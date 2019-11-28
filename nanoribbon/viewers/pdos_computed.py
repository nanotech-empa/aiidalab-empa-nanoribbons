import gzip
import re

import nglview
import gzip
import ipywidgets as ipw
import bqplot as bq
import numpy as np
from io import StringIO
from base64 import b64encode
from xml.etree import ElementTree
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

        self.vacuum_level = workcalc.get_extra('vacuum_level')
        self.homo = workcalc.get_extra('homo')
        self.lumo = workcalc.get_extra('lumo')
        pdos_calc = get_calc_by_label(workcalc, "export_pdos")
        bands_calc = get_calc_by_label(workcalc, "bands")
        self.structure = bands_calc.inputs.structure
        self.ase_struct = self.structure.get_ase()
        self.natoms = len(self.ase_struct)
        self.selected_spin = None
        self.bands = bands_calc.outputs.output_band.get_bands()
        if self.bands.ndim == 2:
            self.bands = self.bands[None,:,:]
            
            
#1            
        atomic_proj_xml = pdos_calc.outputs.retrieved.open("atomic_proj.xml").name

        root = ElementTree.parse(atomic_proj_xml).getroot()
        self.nbands = int(root.find('HEADER/NUMBER_OF_BANDS').text)
        self.nkpoints = int(root.find('HEADER/NUMBER_OF_K-POINTS').text)
        self.nspins = int(root.find('HEADER/NUMBER_OF_SPIN_COMPONENTS').text)
        self.natwfcs = int(root.find('HEADER/NUMBER_OF_ATOMIC_WFC').text)

        self.kpoint_weights = np.fromstring(root.find('WEIGHT_OF_K-POINTS').text, sep=' ')

        self.eigvalues = np.zeros((self.nspins, self.nbands, self.nkpoints))
        for i in range(self.nspins):
            for k in range(self.nkpoints):
                eigtag = 'EIG.%s'%(i+1) if self.nspins > 1 else 'EIG'
                arr = np.fromstring(root.find('EIGENVALUES/K-POINT.%d/%s'%(k+1, eigtag)).text, sep='\n')
                self.eigvalues[i, :, k] = arr * 13.60569806589 - self.vacuum_level # convert Ry to eV

        self.projections = np.zeros((self.nspins, self.nbands, self.nkpoints, self.natwfcs))
        for i in range(self.nspins):
            for k in range(self.nkpoints):
                for l in range(self.natwfcs):
                    spintag = 'SPIN.%d/'%(i+1) if self.nspins > 1 else ""
                    raw = root.find('PROJECTIONS/K-POINT.%d/%sATMWFC.%d'%(k+1, spintag, l+1)).text
                    arr = np.fromstring(raw.replace(",", "\n"), sep="\n")
                    arr2 = arr.reshape(self.nbands, 2) # group real and imaginary part together
                    arr3 = np.sum(np.square(arr2), axis=1) # calculate square of abs value
                    self.projections[i, :, k, l] = arr3            
            
            
#2            
        output_log = pdos_calc.outputs.retrieved.open('aiida.out').name

        # parse mapping atomic functions -> atoms
        # example:     state #   2: atom   1 (C  ), wfc  2 (l=1 m= 1)
        content = open(output_log).read()
        m = re.findall("\n\s+state #\s*(\d+): atom\s*(\d+) ", content, re.DOTALL)
        self.atmwfc2atom = dict([(int(i), int(j)) for i,j in m])
        assert len(self.atmwfc2atom) == self.natwfcs
        assert len(set(self.atmwfc2atom.values())) == self.natoms  
    
    
#3
        self.kpts = np.linspace(0.0, 0.5, self.nkpoints)
        #atmwfcsues, projections = correct_band_crossings(kpts, eigvalues, projections)

        self.bands = np.swapaxes(self.eigvalues, 1, 2) + self.vacuum_level


####BOH

        style = {"description_width":"200px"}
        layout = ipw.Layout(width="600px")
        self.sigma_slider = ipw.FloatSlider(description="Broadening [eV]", min=0.01, max=0.5, value=0.1, step=0.01,
                                       continuous_update=False, layout=layout, style=style)
        #sigma_slider.observe(on_change, names='value')
        self.ngauss_slider = ipw.IntSlider(description="Methfessel-Paxton order", min=0, max=3, value=0,
                                      continuous_update=False, layout=layout, style=style)
        #ngauss_slider.observe(on_change, names='value')

        self.colorpicker = ipw.ColorPicker(concise=True, description='PDOS color', value='orange', style=style)

        center = (self.homo + self.lumo)/2.0
        self.emin_box = ipw.FloatText(description="Emin (eV)", value=np.round(center-3.0, 1), step=0.1, style=style)
        self.emax_box = ipw.FloatText(description="Emax (eV)", value=np.round(center+3.0, 1), step=0.1, style=style)
        self.band_proj_box = ipw.FloatText(description="Max band width (eV)", value=0.1, step=0.01, style=style)


#10
        #def on_change(c):
        #    with plot_out:
        #        clear_output()
        #        plot_all()

        def on_plot_click(c):
            with plot_out:
                clear_output()
                plot_all()


    #on_change(None)        
        
        self.plot_button = ipw.Button(description="Plot")
        self.plot_button.on_click(on_plot_click)

        self.selected_atoms = set()    
        self.viewer = nglview.NGLWidget()

        self.viewer.add_component(nglview.ASEStructure(self.ase_struct)) # adds ball+stick
        self.viewer.add_unitcell()
        self.viewer.center()

        self.viewer.observe(on_picked, names='picked')
        self.plot_out = ipw.Output()

        boxes=[self.sigma_slider, self.ngauss_slider, self.viewer, 
               self.emin_box, self.emax_box, self.band_proj_box, 
               self.colorpicker, self.plot_button, self.plot_out]
        
        super(NanoribbonShowWidget, self).__init__(boxes, **kwargs)

        
#4

    def calc_pdos(self,sigma, ngauss, Emin, Emax, atmwfcs=None):
        DeltaE = 0.01
        x = np.arange(Emin,Emax,DeltaE)

        # calculate histogram for all spins, bands, and kpoints in parallel
        xx = np.tile(x[:, None, None, None], (1, self.nspins, nbands, nkpoints))
        arg = (xx - self.eigvalues) / sigma
        delta = w0gauss(arg, n=ngauss) / sigma
        if atmwfcs:
            p = np.sum(self.projections[:,:,:,atmwfcs], axis=3) # sum over selected atmwfcs
        else:
            p = np.sum(self.projections, axis=3) # sum over all atmwfcs

        c = delta * p * self.kpoint_weights
        y = np.sum(c, axis=(2,3)) # sum over bands and kpoints

        return x, y

#5
    def igor_pdos(self):
        center = (self.homo + self.lumo)/2.0
        Emin, Emax = center-3.0, center+3.0
        if self.selected_atoms:
            atmwfcs = [k-1 for k, v in self.atmwfc2atom.items() if v-1 in self.selected_atoms]
        else:
            atmwfcs = None
        pdos = calc_pdos(ngauss=self.ngauss_slider.value, sigma=self.sigma_slider.value, Emin=Emin, Emax=Emax, atmwfcs=atmwfcs)
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

    def mk_igor_link(self):
        igorvalue = igor_pdos()
        igorfile = b64encode(igorvalue)
        filename = self.ase_struct.get_chemical_formula() + "_pk%d.itx" % self.structure.pk

        html = '<a download="{}" href="'.format(filename)
        html += 'data:chemical/x-igor;name={};base64,{}"'.format(filename, igorfile)
        html += ' id="pdos_link"'
        html += ' target="_blank">Export itx-PDOS</a>'

        javascript = 'var link = document.getElementById("pdos_link");\n'
        javascript += 'link.download = "{}";'.format(filename)

        display(HTML(html))

    def mk_bands_txt_link(self):
        tempio = io.StringIO()
        with tempio as f:
            np.savetxt(f, bands[0])
            value = f.getvalue()

        enc_file = b64encode(value)
        filename = self.ase_struct.get_chemical_formula() + "_pk%d.txt" % self.structure.pk

        html = '<a download="{}" href="'.format(filename)
        html += 'data:chemical/x-igor;name={};base64,{}"'.format(filename, enc_file)
        html += ' id="bands_link"'
        html += ' target="_blank">Export bands .txt</a>'

        javascript = 'var link = document.getElementById("bands_link");\n'
        javascript += 'link.download = "{}";'.format(filename)

        display(HTML(html))

    def mk_png_link(self,fig):
        imgdata = StringIO.StringIO()
        fig.savefig(imgdata, format='png', dpi=300, bbox_inches='tight')
        imgdata.seek(0)  # rewind the data
        pngfile = b64encode(imgdata.buf)

        filename = self.ase_struct.get_chemical_formula() + "_pk%d.png" % self.structure.pk

        html = '<a download="{}" href="'.format(filename)
        html += 'data:image/png;name={};base64,{}"'.format(filename, pngfile)
        html += ' id="pdos_png_link"'
        html += ' target="_blank">Export png</a>'

        display(HTML(html))

    def mk_pdf_link(self,fig):
        imgdata = StringIO.StringIO()
        fig.savefig(imgdata, format='pdf', bbox_inches='tight')
        imgdata.seek(0)  # rewind the data
        pdffile = b64encode(imgdata.buf)

        filename = self.ase_struct.get_chemical_formula() + "_pk%d.pdf" % self.structure.pk

        html = '<a download="{}" href="'.format(filename)
        html += 'data:image/png;name={};base64,{}"'.format(filename, pdffile)
        html += ' id="pdos_png_link"'
        html += ' target="_blank">Export pdf</a>'

        display(HTML(html))
    
#6
    def plot_pdos(self,ax, pdos_full, ispin, pdos=None):
        x, y = pdos_full
        ax.plot(y[:,ispin], x, color='black') # vertical plot
        tfrm = matplotlib.transforms.Affine2D().rotate_deg(90) + ax.transData
        ax.fill_between(x, 0.0, -y[:,ispin], facecolor='lightgray', transform=tfrm)

        ax.set_xlim(0, 1.05*np.amax(y))
        ax.set_xlabel('DOS [a.u.]')
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

        if pdos != None:
            x, y = pdos
            col = matplotlib.colors.to_rgb(self.colorpicker.value)
            ax.plot(y[:,ispin], x, color='k')
            #ax.plot(y[:,ispin], x, color='blue')
            tfrm = matplotlib.transforms.Affine2D().rotate_deg(90) + ax.transData
            ax.fill_between(x, 0.0, -y[:,ispin], facecolor=col, transform=tfrm)
            #ax.fill_between(x, 0.0, -y[:,ispin], facecolor='cyan', transform=tfrm)

#7

    def plot_bands(self,ax, ispin, fig_aspect, atmwfcs=None):
        nspins, nkpoints, nbands = self.bands.shape

        ax.set_title("Spin %d"%ispin)
        ax.axhline(y=homo, linewidth=2, color='gray', ls='--')

        ax.set_xlabel('k [$2\pi/a$]')
        x_data = np.linspace(0.0, 0.5, nkpoints)
        ax.set_xlim(0, 0.5)

        y_datas = self.bands[ispin,:,:] - self.vacuum_level

        for i_band in range(nbands):
            y_data = y_datas[:,i_band]
            ax.plot(x_data, y_data, '-', color='black')

            ### plot the projection on bands
            if atmwfcs is not None:

                line_widths = np.zeros(len(x_data))
                for atomwfc in atmwfcs:
                    line_widths += self.projections[ispin, i_band, :, atomwfc]*self.band_proj_box.value

                edge_up, edge_down = var_width_lines(x_data, y_data, line_widths, fig_aspect)

                edge_up_interp = np.interp(x_data, edge_up[0], edge_up[1])
                edge_down_interp = np.interp(x_data, edge_down[0], edge_down[1])

                conv_kernel = np.ones(3)/3
                edge_up_smooth = scipy.ndimage.filters.convolve(edge_up_interp, conv_kernel)
                edge_down_smooth = scipy.ndimage.filters.convolve(edge_down_interp, conv_kernel)

                #ax.plot(x_data, edge_up_interp, '-', color='orange')
                #ax.plot(x_data, edge_down_interp, '-', color='orange')
                ax.fill_between(x_data, edge_down_smooth, edge_up_smooth, facecolor=matplotlib.colors.to_rgb(self.colorpicker.value))
                #ax.fill_between(x_data, edge_down_smooth, edge_up_smooth, facecolor='cyan')

#8
    def plot_all(self):

        sigma = self.sigma_slider.value
        ngauss = self.ngauss_slider.value
        emin = self.emin_box.value
        emax = self.emax_box.value

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
        if self.selected_atoms:
            # collect all atmwfc located on selected atoms
            atmwfcs = [k-1 for k, v in self.atmwfc2atom.items() if v-1 in self.selected_atoms]
            print("Selected atmwfcs: "+str(atmwfcs))
            pdos = calc_pdos(ngauss=ngauss, sigma=sigma, Emin=emin, Emax=emax, atmwfcs=atmwfcs)

        for ispin in range(self.nspins):
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
    def on_picked(self,c):

        if 'atom1' not in self.viewer.picked.keys():
            return # did not click on atom
        with plot_out:
            clear_output()
            #viewer.clear_representations()
            self.viewer.component_0.remove_ball_and_stick()
            self.viewer.component_0.remove_ball_and_stick()
            self.viewer.add_ball_and_stick()
            #viewer.add_unitcell()

            idx = self.viewer.picked['atom1']['index']

            # toggle
            if idx in self.selected_atoms:
                self.selected_atoms.remove(idx)
            else:
                self.selected_atoms.add(idx)

            #if(selection):
            sel_str = ",".join([str(i) for i in sorted(self.selected_atoms)])
            self.viewer.add_representation('ball+stick', selection="@"+sel_str, color='red', aspectRatio=3.0)
            #else:
            #    print ("nothing selected")
            self.viewer.picked = {} # reset, otherwise immidiately selecting same atom again won't create change event

            #plot_all()
            


            
            


#    @property
#    def spindensity_calc(self):
#        try:
#            return get_calc_by_label(self._workcalc, "export_spinden")
#        except:
#            return None
#
#    def spindensity_2d(self):
#        if self.spindensity_calc:
#            spinden_cube = read_cube(self.spindensity_calc.outputs.retrieved.open("_spin.cube.gz").name)
#            spinden_cube['data'] *= 2000 # normalize scale
#            def on_spinden_plot_change(c):
#                with spinden_out:
#                    clear_output()
#                    fig, ax = plt.subplots()
#                    fig.dpi = 150.0
#                    cax = plot_cube(ax, spinden_cube, 1, 'seismic')
#                    fig.colorbar(cax,  label='arbitrary unit')
#                    plot_overlay_struct(ax, spinden_alpha_slider.value)
#                    plt.show()
#
#            spinden_alpha_slider = ipw.FloatSlider(description="opacity", value=0.5, max=1.0, continuous_update=False)
#            spinden_alpha_slider.observe(on_spinden_plot_change, names='value')
#            spinden_out = ipw.Output()
#            on_spinden_plot_change(None)
#            return ipw.VBox([spinden_out, spinden_alpha_slider])
#        else:
#            print("Could not find spin density")
#
#    def spindensity_3d(self):
#        if self.spindensity_calc:
#            try:
#                ngl_view = nglview.NGLWidget()
#                file_path = self.spindensity_calc.outputs.retrieved.open("_spin_full.cube.gz").name
#                setup_spin_cube_plot(file_path, ngl_view)
#                isosurf_slider = ipw.FloatSlider(
#                    value=1e-3,
#                    min=1e-4,
#                    max=1e-2,
#                    step=1e-4, 
#                    description='isovalue',
#                    readout_format='.1e'
#                )
#                isosurf_slider.observe(lambda c: on_spin_isosurf_change(ngl_view), names='value')
#
#                return ipw.VBox([ngl_view, isosurf_slider])
#
#            except Exception as e:
#                print("Full spin density cube could not be visualized:")
#                print (e.message)