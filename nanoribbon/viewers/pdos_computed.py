import gzip
import re

import nglview
import gzip
import ipywidgets as ipw
import bqplot as bq
import numpy as np
from io import StringIO
from base64 import b64encode
import matplotlib.pyplot as plt
from IPython.display import clear_output
from collections import OrderedDict
import scipy.constants as const
import ase
from ase.data import covalent_radii, atomic_numbers
from ase.data.colors import cpk_colors
from ase.neighborlist import NeighborList
import ase.io
import ase.io.cube
from aiida.orm import CalcJobNode, load_node, QueryBuilder, WorkChainNode

on_band_click_global = None





def get_calc_by_label(workcalc, label):
    calcs = get_calcs_by_label(workcalc, label)
    assert len(calcs) == 1
    return calcs[0]
    
def get_calcs_by_label(workcalc, label):
    qb = QueryBuilder()
    qb.append(WorkChainNode, filters={'uuid':workcalc.uuid})
    qb.append(CalcJobNode, with_incoming=WorkChainNode, filters={'label':label})
    calcs = [ c[0] for c in qb.all() ] 
    for calc in calcs: 
        assert(calc.is_finished_ok == True)
    return calcs    

def w0gauss(x,n):
    arg = np.minimum(200.0, x**2)
    w0gauss = np.exp ( - arg) / np.sqrt(np.pi)
    if n==0 :
        return w0gauss
    hd = 0.0
    hp = np.exp( - arg)
    ni = 0
    a = 1.0 / np.sqrt(np.pi)
    for i in range(1, n+1):
        hd = 2.0 * x * hp - 2.0 * ni * hd
        ni = ni + 1
        a = - a / (i * 4.0)
        hp = 2.0 * x * hd-2.0 * ni * hp
        ni = ni + 1
        w0gauss = w0gauss + a * hp
    return w0gauss

def var_width_lines(x, y, lw, aspect):
    
    nx = len(x)
    edge_up = [np.zeros(nx), np.zeros(nx)]
    edge_down = [np.zeros(nx), np.zeros(nx)]
    
    for i_x in range(nx):
        if i_x == 0:
            dx = x[i_x] - x[i_x+1]
            dy = y[i_x] - y[i_x+1]
        elif i_x == nx - 1:
            dx = x[i_x-1] - x[i_x]
            dy = y[i_x-1] - y[i_x]
        else:
            dx = x[i_x-1] - x[i_x+1]
            dy = y[i_x-1] - y[i_x+1]
            
        line_dir = np.array((dx,  dy))
        # Convert line vector to "figure coordinates"
        line_dir[0] /= aspect
        
        perp_dir = np.array((line_dir[1], -line_dir[0]))
        shift_vec = perp_dir/np.sqrt(perp_dir[0]**2 + perp_dir[1]**2)*lw[i_x]
        
        # convert shift_vec back to "data coordinates"
        shift_vec[0] *= aspect
        
        edge_up[0][i_x] = x[i_x] + shift_vec[0]
        edge_up[1][i_x] = y[i_x] + shift_vec[1]
        edge_down[0][i_x] = x[i_x] - shift_vec[0]
        edge_down[1][i_x] = y[i_x] - shift_vec[1]
    
    return edge_up, edge_down

class NanoribbonPDOSWidget(ipw.HBox):
    def __init__(self, workcalc, **kwargs):
        self._workcalc = workcalc


        self.orbitals_calcs = get_calcs_by_label(workcalc, "export_pdos")
        bands_calc = get_calc_by_label(workcalc, "bands")
        self.structure = bands_calc.inputs.structure
        self.ase_struct = self.structure.get_ase()
        self.natoms = len(self.ase_struct)
        self.selected_cube = None
        self.selected_spin = None
        self.selected_band = None
        self.bands = bands_calc.outputs.output_band.get_bands()
        if self.bands.ndim == 2:
            self.bands = self.bands[None,:,:]
            
            
#1            
        self.atomic_proj_xml = pdos_calc.out.retrieved.get_abs_path('atomic_proj.xml')

        self.root = ElementTree.parse(atomic_proj_xml).getroot()
        self.nbands = int(root.find('HEADER/NUMBER_OF_BANDS').text)
        self.nkpoints = int(root.find('HEADER/NUMBER_OF_K-POINTS').text)
        self.nspins = int(root.find('HEADER/NUMBER_OF_SPIN_COMPONENTS').text)
        self.natwfcs = int(root.find('HEADER/NUMBER_OF_ATOMIC_WFC').text)

        self.kpoint_weights = np.fromstring(self.root.find('WEIGHT_OF_K-POINTS').text, sep=' ')

        self.eigvalues = np.zeros((self.nspins, self.nbands, self.nkpoints))
        for i in range(self.nspins):
            for k in range(self.nkpoints):
                eigtag = 'EIG.%s'%(i+1) if nspins > 1 else 'EIG'
                arr = np.fromstring(root.find('EIGENVALUES/K-POINT.%d/%s'%(k+1, eigtag)).text, sep='\n')
                self.eigvalues[i, :, k] = arr * 13.60569806589 - vacuum_level # convert Ry to eV

        self.projections = np.zeros((nspins, nbands, nkpoints, natwfcs))
        for i in range(nspins):
            for k in range(nkpoints):
                for l in range(natwfcs):
                    spintag = 'SPIN.%d/'%(i+1) if nspins > 1 else ""
                    raw = root.find('PROJECTIONS/K-POINT.%d/%sATMWFC.%d'%(k+1, spintag, l+1)).text
                    arr = np.fromstring(raw.replace(",", "\n"), sep="\n")
                    arr2 = arr.reshape(nbands, 2) # group real and imaginary part together
                    arr3 = np.sum(np.square(arr2), axis=1) # calculate square of abs value
                    self.projections[i, :, k, l] = arr3            
            
            
#2            
        output_log = pdos_calc.out.retrieved.get_abs_path('aiida.out')

        # parse mapping atomic functions -> atoms
        # example:     state #   2: atom   1 (C  ), wfc  2 (l=1 m= 1)
        content = open(output_log).read()
        m = re.findall("\n\s+state #\s*(\d+): atom\s*(\d+) ", content, re.DOTALL)
        self.atmwfc2atom = dict([(int(i), int(j)) for i,j in m])
        assert len(atmwfc2atom) == self.natwfcs
        assert len(set(atmwfc2atom.values())) == self.natoms  
    
    
#3
        self.kpts = np.linspace(0.0, 0.5, self.nkpoints)
        #eigvalues, projections = correct_band_crossings(kpts, eigvalues, projections)

        self.bands = np.swapaxes(self.eigvalues, 1, 2) + self.vacuum_level


####BOH


        self.band_plots = []
        boxes = []
        self.eff_mass_parabolas = []
        for ispin in range(self.bands.shape[0]):
            box, plot, eff_mass_parabola = self.plot_spin(ispin)
            boxes.append(box)
            self.band_plots.append(plot)
            self.eff_mass_parabolas.append(eff_mass_parabola)

        layout = ipw.Layout(padding="5px", margin="0px")
        self.info_out = ipw.Output(layout=layout)
        self.kpnt_out = ipw.Output(layout=layout)
        self.orb_out = ipw.Output(layout=layout)

        layout = ipw.Layout(width="400px")

        ### -----------------------------
        ### Slider to control how many points of the band to use for parabolic fit

        # Odd values of fit have better accuracy, so it's worth it to disable even values
        self.efm_fit_slider = ipw.IntSlider(description="eff. mass fit", min=3, max=15, step=2, continuous_update=False, layout=layout)
        # Only if a band is selected, selecting a new effective mass fit will update the plot and infopanel
        on_efm_fit_change = lambda c: self.on_band_change() if self.selected_spin else None
        self.efm_fit_slider.observe(on_efm_fit_change, names='value')
        ### -----------------------------

        self.kpoint_slider = ipw.IntSlider(description="k-point", min=1, max=1, continuous_update=False, layout=layout)
        self.kpoint_slider.observe(self.on_kpoint_change, names='value')

        self.height_slider = ipw.SelectionSlider(description="height", options={"---":0}, continuous_update=False, layout=layout)
        self.height_slider.observe(self.on_orb_plot_change, names='value')

        self.orb_alpha_slider = ipw.FloatSlider(description="opacity", value=0.5, max=1.0, continuous_update=False, layout=layout)
        self.orb_alpha_slider.observe(self.on_orb_plot_change, names='value')

        self.colormap_slider = ipw.FloatRangeSlider(description='colormap', min=-10, max=-1, step=0.5,
                                       value=[-6, -3], continuous_update=False, readout_format='.1f', layout=layout)
        self.colormap_slider.observe(self.on_orb_plot_change, names='value')
        
        layout = ipw.Layout(align_items="center")
        side_box = ipw.VBox([self.info_out, self.efm_fit_slider, self.kpoint_slider,
                             self.height_slider, self.orb_alpha_slider, self.colormap_slider,
                             self.kpnt_out, self.orb_out], layout=layout)
        boxes.append(side_box)        
        super(NanoribbonShowWidget, self).__init__(boxes, **kwargs)

        
#4

    def calc_pdos(sigma, ngauss, Emin, Emax, atmwfcs=None):
        DeltaE = 0.01
        x = np.arange(Emin,Emax,DeltaE)

        # calculate histogram for all spins, bands, and kpoints in parallel
        xx = np.tile(x[:, None, None, None], (1, nspins, nbands, nkpoints))
        arg = (xx - eigvalues) / sigma
        delta = w0gauss(arg, n=ngauss) / sigma
        if atmwfcs:
            p = np.sum(projections[:,:,:,atmwfcs], axis=3) # sum over selected atmwfcs
        else:
            p = np.sum(projections, axis=3) # sum over all atmwfcs

        c = delta * p * kpoint_weights
        y = np.sum(c, axis=(2,3)) # sum over bands and kpoints

        return x, y

#5
    def igor_pdos():
        center = (homo + lumo)/2.0
        Emin, Emax = center-3.0, center+3.0
        if selected_atoms:
            atmwfcs = [k-1 for k, v in atmwfc2atom.items() if v-1 in selected_atoms]
        else:
            atmwfcs = None
        pdos = calc_pdos(ngauss=ngauss_slider.value, sigma=sigma_slider.value, Emin=Emin, Emax=Emax, atmwfcs=atmwfcs)
        e = pdos[0]
        p = pdos[1].transpose()[0]
        tempio = io.StringIO()
        with tempio as f:
            f.write(u'IGOR\rWAVES\te1\tp1\rBEGIN\r')
            for x, y in zip(e, p):
                f.write(u'\t{:.8f}\t{:.8f}\r'.format(x, y))
            f.write(u'END\r')
            f.write(u'X SetScale/P x 0,1,"", e1; SetScale y 0,0,"", e1\rX SetScale/P x 0,1,"", p1; SetScale y 0,0,"", p1\r')
            return f.getvalue()

    def mk_igor_link():
        igorvalue = igor_pdos()
        igorfile = b64encode(igorvalue)
        filename = ase_struct.get_chemical_formula() + "_pk%d.itx" % structure.pk

        html = '<a download="{}" href="'.format(filename)
        html += 'data:chemical/x-igor;name={};base64,{}"'.format(filename, igorfile)
        html += ' id="pdos_link"'
        html += ' target="_blank">Export itx-PDOS</a>'

        javascript = 'var link = document.getElementById("pdos_link");\n'
        javascript += 'link.download = "{}";'.format(filename)

        display(HTML(html))

    def mk_bands_txt_link():
        tempio = io.StringIO()
        with tempio as f:
            np.savetxt(f, bands[0])
            value = f.getvalue()

        enc_file = b64encode(value)
        filename = ase_struct.get_chemical_formula() + "_pk%d.txt" % structure.pk

        html = '<a download="{}" href="'.format(filename)
        html += 'data:chemical/x-igor;name={};base64,{}"'.format(filename, enc_file)
        html += ' id="bands_link"'
        html += ' target="_blank">Export bands .txt</a>'

        javascript = 'var link = document.getElementById("bands_link");\n'
        javascript += 'link.download = "{}";'.format(filename)

        display(HTML(html))

    def mk_png_link(fig):
        imgdata = StringIO.StringIO()
        fig.savefig(imgdata, format='png', dpi=300, bbox_inches='tight')
        imgdata.seek(0)  # rewind the data
        pngfile = b64encode(imgdata.buf)

        filename = ase_struct.get_chemical_formula() + "_pk%d.png" % structure.pk

        html = '<a download="{}" href="'.format(filename)
        html += 'data:image/png;name={};base64,{}"'.format(filename, pngfile)
        html += ' id="pdos_png_link"'
        html += ' target="_blank">Export png</a>'

        display(HTML(html))

    def mk_pdf_link(fig):
        imgdata = StringIO.StringIO()
        fig.savefig(imgdata, format='pdf', bbox_inches='tight')
        imgdata.seek(0)  # rewind the data
        pdffile = b64encode(imgdata.buf)

        filename = ase_struct.get_chemical_formula() + "_pk%d.pdf" % structure.pk

        html = '<a download="{}" href="'.format(filename)
        html += 'data:image/png;name={};base64,{}"'.format(filename, pdffile)
        html += ' id="pdos_png_link"'
        html += ' target="_blank">Export pdf</a>'

        display(HTML(html))
    
#6
    def plot_pdos(ax, pdos_full, ispin, pdos=None):
        x, y = pdos_full
        ax.plot(y[:,ispin], x, color='black') # vertical plot
        tfrm = matplotlib.transforms.Affine2D().rotate_deg(90) + ax.transData
        ax.fill_between(x, 0.0, -y[:,ispin], facecolor='lightgray', transform=tfrm)

        ax.set_xlim(0, 1.05*np.amax(y))
        ax.set_xlabel('DOS [a.u.]')
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

        if pdos != None:
            x, y = pdos
            col = matplotlib.colors.to_rgb(colorpicker.value)
            ax.plot(y[:,ispin], x, color='k')
            #ax.plot(y[:,ispin], x, color='blue')
            tfrm = matplotlib.transforms.Affine2D().rotate_deg(90) + ax.transData
            ax.fill_between(x, 0.0, -y[:,ispin], facecolor=col, transform=tfrm)
            #ax.fill_between(x, 0.0, -y[:,ispin], facecolor='cyan', transform=tfrm)

#7

    def plot_bands(ax, ispin, fig_aspect, atmwfcs=None):
        nspins, nkpoints, nbands = bands.shape

        ax.set_title("Spin %d"%ispin)
        ax.axhline(y=homo, linewidth=2, color='gray', ls='--')

        ax.set_xlabel('k [$2\pi/a$]')
        x_data = np.linspace(0.0, 0.5, nkpoints)
        ax.set_xlim(0, 0.5)

        y_datas = bands[ispin,:,:] - vacuum_level

        for i_band in range(nbands):
            y_data = y_datas[:,i_band]
            ax.plot(x_data, y_data, '-', color='black')

            ### plot the projection on bands
            if atmwfcs is not None:

                line_widths = np.zeros(len(x_data))
                for atomwfc in atmwfcs:
                    line_widths += projections[ispin, i_band, :, atomwfc]*band_proj_box.value

                edge_up, edge_down = var_width_lines(x_data, y_data, line_widths, fig_aspect)

                edge_up_interp = np.interp(x_data, edge_up[0], edge_up[1])
                edge_down_interp = np.interp(x_data, edge_down[0], edge_down[1])

                conv_kernel = np.ones(3)/3
                edge_up_smooth = scipy.ndimage.filters.convolve(edge_up_interp, conv_kernel)
                edge_down_smooth = scipy.ndimage.filters.convolve(edge_down_interp, conv_kernel)

                #ax.plot(x_data, edge_up_interp, '-', color='orange')
                #ax.plot(x_data, edge_down_interp, '-', color='orange')
                ax.fill_between(x_data, edge_down_smooth, edge_up_smooth, facecolor=matplotlib.colors.to_rgb(colorpicker.value))
                #ax.fill_between(x_data, edge_down_smooth, edge_up_smooth, facecolor='cyan')

#8
    def plot_all():

        sigma = sigma_slider.value
        ngauss = ngauss_slider.value
        emin = emin_box.value
        emax = emax_box.value

        figsize = (12, 8)
        fig = plt.figure()
        fig.set_size_inches(figsize[0], figsize[1])
        fig.subplots_adjust(wspace=0.1, hspace=0)

        fig_aspect = figsize[1]/(figsize[0]/4.0) * 0.5/(emax-emin)

        sharey = None
        pdos_full = calc_pdos(ngauss=ngauss, sigma=sigma, Emin=emin, Emax=emax)

        # DOS projected to selected atoms
        pdos = None
        atmwfcs = None
        if selected_atoms:
            # collect all atmwfc located on selected atoms
            atmwfcs = [k-1 for k, v in atmwfc2atom.items() if v-1 in selected_atoms]
            print("Selected atmwfcs: "+str(atmwfcs))
            pdos = calc_pdos(ngauss=ngauss, sigma=sigma, Emin=emin, Emax=emax, atmwfcs=atmwfcs)

        for ispin in range(nspins):
            # band plot
            ax1 = fig.add_subplot(1, 4, 2*ispin+1, sharey=sharey)
            if not sharey:
                ax1.set_ylabel('E [eV]')
                sharey = ax1
            else:
                ax1.tick_params(axis='y', which='both',left='on',right='off', labelleft='off')
            plot_bands(ax=ax1, ispin=ispin, fig_aspect=fig_aspect, atmwfcs=atmwfcs)

            # pdos plot
            ax2 = fig.add_subplot(1, 4, 2*ispin+2, sharey=sharey)
            ax2.tick_params(axis='y', which='both',left='on',right='off', labelleft='off')
            plot_pdos(ax=ax2, pdos_full=pdos_full, ispin=ispin, pdos=pdos)

        sharey.set_ylim(emin, emax)

        plt.show()  

        mk_png_link(fig)
        mk_pdf_link(fig)
        mk_bands_txt_link()
        mk_igor_link()
        
        
#9
    def on_picked(c):
        global selected_atoms

        if 'atom1' not in viewer.picked.keys():
            return # did not click on atom
        with plot_out:
            clear_output()
            #viewer.clear_representations()
            viewer.component_0.remove_ball_and_stick()
            viewer.component_0.remove_ball_and_stick()
            viewer.add_ball_and_stick()
            #viewer.add_unitcell()

            idx = viewer.picked['atom1']['index']

            # toggle
            if idx in selected_atoms:
                selected_atoms.remove(idx)
            else:
                selected_atoms.add(idx)

            #if(selection):
            sel_str = ",".join([str(i) for i in sorted(selected_atoms)])
            viewer.add_representation('ball+stick', selection="@"+sel_str, color='red', aspectRatio=3.0)
            #else:
            #    print ("nothing selected")
            viewer.picked = {} # reset, otherwise immidiately selecting same atom again won't create change event

            #plot_all()
            

#10
    #def on_change(c):
    #    with plot_out:
    #        clear_output()
    #        plot_all()

    def on_plot_click(c):
        with plot_out:
            clear_output()
            plot_all()

    style = {"description_width":"200px"}
    layout = ipw.Layout(width="600px")
    sigma_slider = ipw.FloatSlider(description="Broadening [eV]", min=0.01, max=0.5, value=0.1, step=0.01,
                                   continuous_update=False, layout=layout, style=style)
    #sigma_slider.observe(on_change, names='value')
    ngauss_slider = ipw.IntSlider(description="Methfessel-Paxton order", min=0, max=3, value=0,
                                  continuous_update=False, layout=layout, style=style)
    #ngauss_slider.observe(on_change, names='value')

    colorpicker = ipw.ColorPicker(concise=True, description='PDOS color', value='orange', style=style)

    center = (homo + lumo)/2.0
    emin_box = ipw.FloatText(description="Emin (eV)", value=np.round(center-3.0, 1), step=0.1, style=style)
    emax_box = ipw.FloatText(description="Emax (eV)", value=np.round(center+3.0, 1), step=0.1, style=style)
    band_proj_box = ipw.FloatText(description="Max band width (eV)", value=0.1, step=0.01, style=style)


    plot_button = ipw.Button(description="Plot")
    plot_button.on_click(on_plot_click)

    selected_atoms = set()    
    viewer = nglview.NGLWidget()

    viewer.add_component(nglview.ASEStructure(ase_struct)) # adds ball+stick
    viewer.add_unitcell()
    viewer.center()

    viewer.observe(on_picked, names='picked')
    plot_out = ipw.Output()

    display(sigma_slider, ngauss_slider, viewer, emin_box, emax_box, band_proj_box, colorpicker, plot_button, plot_out)
    #on_change(None)
            
            
########STOP            
            
    def plot_spin(self, ispin):
        global on_band_click_global
        _, nkpoints, nbands = self.bands.shape
        homo = self._workcalc.get_extra('homo')
        lumo = self._workcalc.get_extra('lumo')
        
        center = (homo + lumo) / 2.0
        x_sc = bq.LinearScale()
        y_sc = bq.LinearScale(min=center-3.0, max=center+3.0, )

        color_sc = bq.ColorScale(colors=['gray', 'red'], min=0.0, max=1.0)
        colors = np.zeros(nbands)

        Lx = self.structure.cell_lengths[0]
        x_max = np.pi / Lx
        ax_x = bq.Axis(label=u'kA^-1', scale=x_sc, grid_lines='solid', tick_format='.3f', tick_values=[0, x_max]) #, tick_values=[0.0, 0.5])
        ax_y = bq.Axis(label='eV', scale=y_sc, orientation='vertical', grid_lines='solid')

        x_data = np.linspace(0.0, x_max, nkpoints)
        y_datas = self.bands[ispin,:,:].transpose() - self._workcalc.get_extra('vacuum_level')

        lines = bq.Lines(x=x_data, y=y_datas, color=colors, animate=True,
                         scales={'x': x_sc, 'y': y_sc, 'color': color_sc})

        homo_line = bq.Lines(x=[0, x_max], y=[homo, homo], line_style='dashed', colors=['red'], scales={'x': x_sc, 'y': y_sc})

        # Initialize the parabola as a random line and set visible to false
        # Later, when it is correctly set, show it.
        eff_mass_parabola = bq.Lines(x=[0, 0], y=[0, 0], visible=False, stroke_width=1.0,
                                     line_style='solid', colors=['blue'], scales={'x': x_sc, 'y': y_sc})

        ratio = 0.25
        
        fig = bq.Figure(axes=[ax_x, ax_y], marks=[lines, homo_line, eff_mass_parabola], title='Spin {}'.format(ispin), 
                        layout=ipw.Layout(height="800px", width="200px"),
                        fig_margin={"left": 45, "top": 60, "bottom":60, "right":40},
                        min_aspect_ratio=ratio, max_aspect_ratio=ratio)

        on_band_click_global = self.on_band_change
        def on_band_click(self, target):
            global on_band_click_global      
            #self.selected_spin = ispin
            #self.selected_band = target['data']['index']
            on_band_click_global(ispin, target['data']['index'])

        lines.on_element_click(on_band_click)

        save_btn = ipw.Button(description="Download png")
        save_btn.on_click(lambda b: fig.save_png()) # save_png() does not work with unicode labels

        igor_link = self.mk_igor_link(ispin)

        box = ipw.VBox([fig, save_btn, igor_link],
                       layout=ipw.Layout(align_items="center", padding="5px", margin="0px"))
        return box, lines, eff_mass_parabola

    def mk_igor_link(self, ispin):
        igorvalue = self.igor_bands(ispin)
        igorfile = b64encode(igorvalue.encode())
        filename = self.ase_struct.get_chemical_formula() + "_bands_spin{}_pk{}.itx".format(ispin, self.structure.id)

        html = '<a download="{}" href="'.format(filename)
        html += 'data:chemical/x-igor;name={};base64,{}"'.format(filename, igorfile)
        html += ' id="pdos_link"'
        html += ' target="_blank">Export itx-Bands</a>'

        return ipw.HTML(html)

    def igor_bands(self, ispin):
        _, nkpoints, nbands = self.bands.shape
        k_axis = np.linspace(0.0, np.pi / self.structure.cell_lengths[0], nkpoints)
        testio = StringIO()
        tosave = self.bands[ispin,:,:].transpose() - self._workcalc.get_extra('vacuum_level')

        with testio as f:
            f.write(u'IGOR\r')
            f.write(u'WAVES')
            f.write(u'\tx1'+(u'\ty{}'*nbands).format(*[x for x in range(nbands)])+u'\r')
            f.write(u'BEGIN\r')
            for i in range(nkpoints):
                f.write(u"\t{:.7f}".format(k_axis[i])) # first column k_axis
                f.write((u"\t{:.7f}"*nbands).format(*tosave[:,i])) # other columns the bands
                f.write(u"\r")
            f.write(u"END\r")
            f.write(u'X SetScale/P x {},{},"", x1; SetScale y 0,0,"", x1\r'.format(0, k_axis[1]-k_axis[0]))
            for idk in range(nbands):
                f.write((u'X SetScale/P x 0,1,"", y{0}; SetScale y 0,0,"", y{0}\r').format(str(idk)))
            return testio.getvalue()

    def on_band_change(self, selected_spin=None, selected_band=None):
        self.selected_spin = selected_spin
        self.selected_band = selected_band
        nspins, _, nbands = self.bands.shape

        with self.info_out:
            clear_output()
            print("selected spin: {}".format(self.selected_spin))
            print("selected band: {}".format(self.selected_band))

            colors = np.zeros((nspins, nbands))
            colors[self.selected_spin, self.selected_band] = 1.0
            for ispin in range(nspins):
                self.band_plots[ispin].color = colors[ispin,:]

            # orbitals_calcs might use fewer nkpoints than bands_calc
            print(self.orbitals_calcs[0].inputs)
            prev_calc = self.orbitals_calcs[0].inputs.parent_folder.creator
            nkpoints_lowres = prev_calc.res.number_of_k_points

            print(nkpoints_lowres, self.selected_spin)
            lower = nkpoints_lowres * self.selected_spin
            upper = lower + nkpoints_lowres
            self.selected_cube_files = []
            for fn in sorted([ fdr.name for orbitals_calc in self.orbitals_calcs
                              for fdr in orbitals_calc.outputs.retrieved.list_objects()]):
                m = re.match("aiida.filplot_K(\d\d\d)_B(\d\d\d)_orbital.cube.gz", fn)
                if not m:
                    continue
                k, b = int(m.group(1)), int(m.group(2))
                if b != self.selected_band + 1:
                    continue
                if lower < k and k <= upper:
                    self.selected_cube_files.append(fn)

            n = len(self.selected_cube_files)
            self.kpoint_slider.max = max(n, 1)
            print("found {} cube files".format(n))
            self.on_kpoint_change(None)

            ### -------------------------------------------
            ### Effective mass calculation and parabola plotting

            meff, parabola_fit, fit_kvals, fit_energies = self.calc_effective_mass(ispin=self.selected_spin)
            print("effective mass: %f"%meff)

            parab_k_arr = np.linspace(np.min(fit_kvals), np.max(fit_kvals), 20)
            parab_e_arr = parabola_fit[0]*parab_k_arr**2 + parabola_fit[1]*parab_k_arr + parabola_fit[2]
            self.eff_mass_parabolas[self.selected_spin].x = parab_k_arr
            self.eff_mass_parabolas[self.selected_spin].y = parab_e_arr
            self.eff_mass_parabolas[self.selected_spin].visible = True

            if nspins > 1:
                self.eff_mass_parabolas[(self.selected_spin+1)%2].visible = False

            ### -------------------------------------------
    def on_kpoint_change(self, c):
        with self.kpnt_out:
            clear_output()
            i = self.kpoint_slider.value
            if i > len(self.selected_cube_files):
                print("Found no cube files")
                self.selected_cube = None
                self.height_slider.options = {"---":0}

            else:    
                fn = self.selected_cube_files[i-1]
                for orbitals_calc in self.orbitals_calcs:
                    try:
                        absfn = orbitals_calc.outputs.retrieved.open(fn).name
                    except FileNotFoundError:
                        continue

                    self.selected_cube = read_cube(absfn)
                    nz = self.selected_cube['data'].shape[2]
                    z0 = self.selected_cube['z0']
                    dz = self.selected_cube['dz']

                    zmid = self.structure.cell_lengths[2] / 2.0
                    options = OrderedDict()
                    for i in range(nz):
                        z = (z0 + dz*i) * 0.529177 - zmid
                        options[u"{:.3f} Å".format(z)] = i
                    self.height_slider.options = options
                    break

                self.on_orb_plot_change(None) 

    def on_orb_plot_change(self, c):
        with self.orb_out:
            clear_output()
            if self.selected_cube is None:
                return

            fig, ax = plt.subplots()
            fig.dpi = 150.0
            vmin = 10 ** self.colormap_slider.value[0]
            vmax = 10 ** self.colormap_slider.value[1]

            cax = plot_cube(ax, self.selected_cube, self.height_slider.value, 'gray', vmin, vmax)
            fig.colorbar(cax, label='e/bohr^3', ticks=[vmin, vmax], format='%.0e', orientation='horizontal', shrink=0.3)

            self.plot_overlay_struct(ax, self.orb_alpha_slider.value)
            plt.show()

    def plot_overlay_struct(self, ax, alpha):
        if alpha == 0:
            return

        # plot overlayed structure
        s = self.ase_struct.repeat((2,1,1))
        cov_radii = [covalent_radii[a.number] for a in s]
        nl = NeighborList(cov_radii, bothways = True, self_interaction = False)
        nl.update(s)

        for at in s:
            #circles
            x, y, z = at.position
            n = atomic_numbers[at.symbol]
            ax.add_artist(plt.Circle((x,y), covalent_radii[n]*0.5, color=cpk_colors[n], fill=True, clip_on=True, alpha=alpha))
            #bonds
            nlist = nl.get_neighbors(at.index)[0]
            for theneig in nlist:
                x,y,z = (s[theneig].position +  at.position)/2
                x0,y0,z0 = at.position
                if (x-x0)**2 + (y-y0)**2 < 2 :
                    ax.plot([x0,x],[y0,y],color=cpk_colors[n],linewidth=2,linestyle='-', alpha=alpha)

    def calc_effective_mass(self, ispin):
        # m* = hbar^2*[d^2E/dk^2]^-1
        hbar = const.value('Planck constant over 2 pi in eV s')
        el_mass = const.m_e*1e-20/const.eV # in eV*s^2/ang^2
        _, nkpoints, _ = self.bands.shape
        band = self.bands[ispin].transpose()[self.selected_band] - self._workcalc.get_extra('vacuum_level')
        k_axis = np.linspace(0.0, np.pi / self.structure.cell_lengths[0], nkpoints)

        num_fit_points = self.efm_fit_slider.value

        if np.amax(band) >= self._workcalc.get_extra('lumo'):
            # conduction band, let's search for effective electron mass (lowest point in energy)
            parabola_ind = np.argmin(band)
        else:
            # valence band, effective hole mass (highest point in energy)
            parabola_ind = np.argmax(band)

        # extend band and k values to neighbouring regions
        band_ext = np.concatenate([np.flip(band, 0)[:-1], band, np.flip(band, 0)[1:]])
        k_vals_ext = np.concatenate([-np.flip(k_axis, 0)[:-1], k_axis, k_axis[-1] + k_axis[1:]])

        # define fitting region
        i_min = parabola_ind - int(np.ceil(num_fit_points/2.0)) + len(band)
        i_max = parabola_ind + int(np.floor(num_fit_points/2.0)) + len(band)

        fit_energies = band_ext[i_min:i_max]
        fit_kvals = k_vals_ext[i_min:i_max]

        #print(k_axis[parabola_ind], band[parabola_ind])
        #print(fit_kvals)
        #print(fit_energies)

        parabola_fit = np.polyfit(fit_kvals, fit_energies, 2)

        meff = hbar**2/(2*parabola_fit[0])/el_mass

        # restrict fitting values to "main region"
        main_region_mask = (fit_kvals >= k_axis[0]) & (fit_kvals <= k_axis[-1])
        fit_energies = fit_energies[main_region_mask]
        fit_kvals = fit_kvals[main_region_mask]

        return meff, parabola_fit, fit_kvals, fit_energies

    @property
    def spindensity_calc(self):
        try:
            return get_calc_by_label(self._workcalc, "export_spinden")
        except:
            return None

    def spindensity_2d(self):
        if self.spindensity_calc:
            spinden_cube = read_cube(self.spindensity_calc.outputs.retrieved.open("_spin.cube.gz").name)
            spinden_cube['data'] *= 2000 # normalize scale
            def on_spinden_plot_change(c):
                with spinden_out:
                    clear_output()
                    fig, ax = plt.subplots()
                    fig.dpi = 150.0
                    cax = plot_cube(ax, spinden_cube, 1, 'seismic')
                    fig.colorbar(cax,  label='arbitrary unit')
                    plot_overlay_struct(ax, spinden_alpha_slider.value)
                    plt.show()

            spinden_alpha_slider = ipw.FloatSlider(description="opacity", value=0.5, max=1.0, continuous_update=False)
            spinden_alpha_slider.observe(on_spinden_plot_change, names='value')
            spinden_out = ipw.Output()
            on_spinden_plot_change(None)
            return ipw.VBox([spinden_out, spinden_alpha_slider])
        else:
            print("Could not find spin density")

    def spindensity_3d(self):
        if self.spindensity_calc:
            try:
                ngl_view = nglview.NGLWidget()
                file_path = self.spindensity_calc.outputs.retrieved.open("_spin_full.cube.gz").name
                setup_spin_cube_plot(file_path, ngl_view)
                isosurf_slider = ipw.FloatSlider(
                    value=1e-3,
                    min=1e-4,
                    max=1e-2,
                    step=1e-4, 
                    description='isovalue',
                    readout_format='.1e'
                )
                isosurf_slider.observe(lambda c: on_spin_isosurf_change(ngl_view), names='value')

                return ipw.VBox([ngl_view, isosurf_slider])

            except Exception as e:
                print("Full spin density cube could not be visualized:")
                print (e.message)