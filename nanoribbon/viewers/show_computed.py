import gzip
import re
import io
import tempfile

import nglview
import ipywidgets as ipw
import bqplot as bq
import numpy as np
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
from traitlets import dlink, observe, Float, Instance, Int

# AiiDA imports.
from aiida.orm import load_node, CalcJobNode, QueryBuilder, WorkChainNode
from aiida.common import exceptions
from aiida.plugins import DataFactory

# AiiDA data objects.
ArrayData = DataFactory("array")
BandsData = DataFactory("array.bands")
StructureData = DataFactory("structure")

ANG_2_BOHR = 1.889725989

def plot_cube(ax, cube_data, cube_atoms, z, cmap, vmin=-1, vmax=+1):
    
    a = np.flip(cube_data[:,:,z].transpose(), axis=0)
    aa = np.tile(a, (1, 2))
    x2 = cube_atoms.cell[0][0]*2.0#['dx'] * aa.shape[1] * 0.529177
    y2 = cube_atoms.cell[1][1]#['dy'] * aa.shape[0] * 0.529177
    
    ax.set_xlabel(u'Å')
    ax.set_ylabel(u'Å')
    ax.set_xlim(0, x2)
    ax.set_ylim(0, y2)
    
    cax = ax.imshow(aa, extent=[0,x2,0,y2], cmap=cmap, vmin=vmin, vmax=vmax)
    return cax

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

def set_cube_isosurf(isovals, colors, ngl_viewer):    
    if hasattr(ngl_viewer, 'component_1'):
        c2 = ngl_viewer.component_1
        c2.clear()
        for isov, col in zip(isovals, colors):
            c2.add_surface(color=col, isolevelType="value", isolevel=isov)

def setup_cube_plot(data, atoms, ngl_viewer):
    atoms.pbc=True
    n_repeat = 2
    atoms_xn = atoms.repeat((n_repeat,1,1))
    data_xn = np.tile(data, (n_repeat,1,1))
    c1 = ngl_viewer.add_component(nglview.ASEStructure(atoms_xn))
    with tempfile.NamedTemporaryFile(mode='w') as tempf:
        ase.io.cube.write_cube(tempf, atoms_xn, data_xn)
        c2 = ngl_viewer.add_component(tempf.name, ext='cube')
        c2.clear()
    

class BandsViewerWidget(ipw.VBox):
    bands = Instance(BandsData, allow_none=True)
    nelectrons = Int(allow_none=True)
    homo = Float(allow_none=True)
    lumo = Float(allow_none=True)
    gap = Float(allow_none=True)
    vacuum_level = Float(0)
    structure = Instance(StructureData, allow_none=True)
    selected_band = Int(allow_none=True)

    def __init__(self, **kwargs):
        self.bands = kwargs['bands']
        self.structure = kwargs['structure']
        self.vacuum_level = kwargs['vacuum_level']
        self.bands_array = self.bands.get_bands()
        self.band_plots = []
        self.nelectrons = kwargs['nelectrons']

        self.gap = kwargs['gap']
        self.homo = kwargs['homo']
        self.lumo = kwargs['lumo']
        
        # Always make the array 3-dimensional.
        if self.bands_array.ndim == 2:
            self.bands_array = self.bands_array[None,:,:]
        self.eff_mass_parabolas = []
        
        layout = ipw.Layout(padding="5px", margin="0px")
        self.my_info_out = ipw.Output(layout=layout)
        
        # Slider to control how many points of the band to use for parabolic fit.
        self.efm_fit_slider = ipw.IntSlider(description="eff. mass fit", min=3, max=15, step=2, continuous_update=False, layout=layout)
        self.band_selector = ipw.IntSlider(description="Band", value=int(self.nelectrons/2) -1, min=0, max=self.bands_array.shape[2], step=1, continuous_update=False, layout=layout)
        self.spin_selector = ipw.RadioButtons(
            options=[('up', 0), ('down', 1)],
            description='Select spin',
            disabled=False
        )


        dlink((self.band_selector, 'value'), (self, 'selected_band'))
        boxes = [self.my_info_out, self.efm_fit_slider, self.band_selector, self.spin_selector]

        plots = []
        for ispin in range(self.bands_array.shape[0]):
            box, plot, eff_mass_parabola = self.plot_bands(ispin)
            plots.append(box)
            self.band_plots.append(plot)
            self.eff_mass_parabolas.append(eff_mass_parabola)
        boxes.append(ipw.HBox(plots))

        # Display the orbital map also initially.
        self.on_band_change(_=None)

        super().__init__(boxes, **kwargs)

    def plot_bands(self, ispin):
        _, nkpoints, nbands = self.bands_array.shape        
        center = (self.homo + self.lumo) / 2.0
        x_sc = bq.LinearScale()
        y_sc = bq.LinearScale(min=center-3.0, max=center+3.0, )

        x_max = np.pi / self.structure.cell_lengths[0]


        x_data = np.linspace(0.0, x_max, nkpoints)
        y_datas = self.bands_array[ispin,:,:].transpose() - self.vacuum_level

        lines = bq.Lines(x=x_data, y=y_datas, color=np.zeros(nbands), animate=True, stroke_width=4.0, scales={'x': x_sc, 'y': y_sc, 'color': bq.ColorScale(colors=['gray', 'red'], min=0.0, max=1.0)})

        homo_line = bq.Lines(x=[0, x_max], y=[self.homo, self.homo], line_style='dashed', colors=['red'], scales={'x': x_sc, 'y': y_sc})

        # Initialize the parabola as a random line and set visible to false
        # Later, when it is correctly set, show it.
        eff_mass_parabola = bq.Lines(x=[0, 0], y=[0, 0], visible=False, stroke_width=1.0,
                                     line_style='solid', colors=['blue'], scales={'x': x_sc, 'y': y_sc})

        ratio = 0.25
        
        ax_x = bq.Axis(label=u'kA^-1', scale=x_sc, grid_lines='solid', tick_format='.3f', tick_values=[0, x_max])
        ax_y = bq.Axis(label='eV', scale=y_sc, orientation='vertical', grid_lines='solid')
        
        fig = bq.Figure(axes=[ax_x, ax_y], marks=[lines, homo_line, eff_mass_parabola], title='Spin {}'.format(ispin), 
                        layout=ipw.Layout(height="800px", width="200px"),
                        fig_margin={"left": 45, "top": 60, "bottom":60, "right":40},
                        min_aspect_ratio=ratio, max_aspect_ratio=ratio)

        save_btn = ipw.Button(description="Download png")
        save_btn.on_click(lambda b: fig.save_png()) # save_png() does not work with unicode labels

        igor_link = self.mk_igor_link(ispin)

        box = ipw.VBox([fig, save_btn, igor_link],
                       layout=ipw.Layout(align_items="center", padding="5px", margin="0px"))
        return box, lines, eff_mass_parabola

    def mk_igor_link(self, ispin):
        igorvalue = self.igor_bands(ispin)
        igorfile = b64encode(igorvalue.encode())
        filename = self.structure.get_ase().get_chemical_formula() + "_bands_spin{}_pk{}.itx".format(ispin, self.structure.id)

        html = '<a download="{}" href="'.format(filename)
        html += 'data:chemical/x-igor;name={};base64,{}"'.format(filename, igorfile)
        html += ' id="pdos_link"'
        html += ' target="_blank">Export itx-Bands</a>'
        return ipw.HTML(html)

    def igor_bands(self, ispin):
        _, nkpoints, nbands = self.bands_array.shape
        k_axis = np.linspace(0.0, np.pi / self.structure.cell_lengths[0], nkpoints)
        testio = io.StringIO()
        tosave = self.bands_array[ispin,:,:].transpose() - self.vacuum_level

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

    @observe('selected_band')
    def on_band_change(self, _=None):
        self.selected_spin = self.spin_selector.value
        nspins, _, nbands = self.bands_array.shape

        with self.my_info_out:
            clear_output()
            print("selected spin: {}".format(self.selected_spin))
            print("selected band: {}".format(self.selected_band))

            colors = np.zeros((nspins, nbands))
            colors[self.selected_spin, self.selected_band] = 1.0

            for ispin in range(nspins):
                self.band_plots[ispin].color = colors[ispin,:]

    def calc_effective_mass(self, ispin):
        # m* = hbar^2*[d^2E/dk^2]^-1
        hbar = const.value('Planck constant over 2 pi in eV s')
        el_mass = const.m_e*1e-20/const.eV # in eV*s^2/ang^2
        _, nkpoints, _ = self.bands_array.shape
        band = self.bands_array[ispin].transpose()[self.selected_band] - self.vacuum_level
        k_axis = np.linspace(0.0, np.pi / self.structure.cell_lengths[0], nkpoints)

        num_fit_points = self.efm_fit_slider.value

        if np.amax(band) >= self.lumo:
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

        parabola_fit = np.polyfit(fit_kvals, fit_energies, 2)

        meff = hbar**2/(2*parabola_fit[0])/el_mass

        # restrict fitting values to "main region"
        main_region_mask = (fit_kvals >= k_axis[0]) & (fit_kvals <= k_axis[-1])
        fit_energies = fit_energies[main_region_mask]
        fit_kvals = fit_kvals[main_region_mask]

        return meff, parabola_fit, fit_kvals, fit_energies


class NanoribbonShowWidget(ipw.HBox):
    def __init__(self, workcalc, **kwargs):
        self._workcalc = workcalc

        print("WorkCalculation PK: {}".format(workcalc.id))
        print("total energy: {} eV".format(workcalc.extras['total_energy']))
        print("gap: {} eV".format(workcalc.extras['gap']))
        print("total magnetization/A: {}".format(workcalc.extras['absolute_magnetization_per_angstr']))
        print("abs. magnetization/A: {}".format(workcalc.extras['total_magnetization_per_angstr']))

        self.orbitals_calcs = get_calcs_by_label(workcalc, "export_orbitals")
        bands_calc = get_calc_by_label(workcalc, "bands")
        self.structure = bands_calc.inputs.structure
        self.selected_cube_files = []
        self.selected_data = None
        self.bands_viewer = BandsViewerWidget(
            bands=bands_calc.outputs.output_band,
            nelectrons=int(bands_calc.outputs.output_parameters['number_of_electrons']),
            vacuum_level=self._workcalc.get_extra('vacuum_level'),
            structure=self.structure,
            homo = self._workcalc.get_extra('homo'),
            lumo = self._workcalc.get_extra('lumo'),
            gap = self._workcalc.get_extra('gap'),
        )
        self.bands_viewer.observe(self.on_band_change, names='selected_band')
        boxes = [self.bands_viewer]

        layout = ipw.Layout(padding="5px", margin="0px")
        self.info_out = ipw.Output(layout=layout)
        self.kpnt_out = ipw.Output(layout=layout)
        self.orb_out = ipw.Output(layout=layout)

        layout = ipw.Layout(width="400px")

        self.kpoint_slider = ipw.IntSlider(description="k-point", min=1, max=1, continuous_update=False, layout=layout)
        self.kpoint_slider.observe(self.on_kpoint_change, names='value')
        self.height_slider = ipw.SelectionSlider(description="height", options={"---":0}, continuous_update=False, layout=layout)
        self.height_slider.observe(self.on_orb_plot_change, names='value')
        self.orb_alpha_slider = ipw.FloatSlider(description="opacity", value=0.5, max=1.0, continuous_update=False, layout=layout)
        self.orb_alpha_slider.observe(self.on_orb_plot_change, names='value')
        self.colormap_slider = ipw.FloatLogSlider(value=0.01,base=10,min=-4, max=-1, step=0.5,
                                       description='Color max',readout_format='.1e', continuous_update=False, layout=layout)
        self.colormap_slider.observe(self.on_orb_plot_change, names='value')
        
        # Orbital 3d visualization.
        self.orbital_ngl = nglview.NGLWidget()
        self.orb_isosurf_slider = ipw.FloatSlider(continuous_update=False,
                    value=1e-3,
                    min=1e-4,
                    max=1e-2,
                    step=1e-4, 
                    description='isovalue',
                    readout_format='.1e'
                )
        self.orb_isosurf_slider.observe(lambda c: set_cube_isosurf([c['new']], ['red'], self.orbital_ngl), names='value')

        self.on_band_change()
        layout = ipw.Layout(align_items="center")
        side_box = ipw.VBox([self.info_out, 
                             self.kpoint_slider,
                             self.height_slider, 
                             self.orb_alpha_slider, 
                             self.colormap_slider,
                             self.kpnt_out, 
                             ipw.HBox([self.orb_out, ipw.VBox([self.orbital_ngl,
                                                               self.orb_isosurf_slider])]
                                     )], layout=layout)
        boxes.append(side_box)        

        super().__init__(boxes, **kwargs)

    def on_band_change(self, _=None):
        self.selected_band = self.bands_viewer.selected_band
        bands = self.bands_viewer.bands_array
        nspins, _, nbands = bands.shape

        with self.info_out:
            clear_output()


            # orbitals_calcs might use fewer nkpoints than bands_calc
            prev_calc = self.orbitals_calcs[0].inputs.parent_folder.creator
            nkpoints_lowres = prev_calc.res.number_of_k_points

            lower = nkpoints_lowres * self.bands_viewer.spin_selector.value
            upper = lower + nkpoints_lowres
            self.selected_cube_files = []
            
            list_of_calcs = []
            for orbitals_calc in self.orbitals_calcs:
                if any(['output_data_multiple' in x for x in orbitals_calc.outputs]):
                    list_of_calcs += [(x, orbitals_calc) for x in orbitals_calc.outputs]
                else:
                    list_of_calcs +=  [(x.name, orbitals_calc) for x in orbitals_calc.outputs.retrieved.list_objects()]
            for (fn, orbitals_calc) in sorted(list_of_calcs, key=lambda x:x[0]):
                m = re.match(".*_K(\d\d\d)_B(\d\d\d).*", fn)
                if m:
                    k, b = int(m.group(1)), int(m.group(2))
                    if b != self.selected_band + 1:
                        continue
                    if lower < k and k <= upper:
                        if fn.startswith('output_data_multiple'):
                            self.selected_cube_files.append(orbitals_calc.outputs[fn])
                        else:
                            self.selected_cube_files.append(orbitals_calc.outputs.retrieved.open(fn).name)

            n = len(self.selected_cube_files)
            self.kpoint_slider.max = max(n, 1)
            print("found {} cube files".format(n))
            self.on_kpoint_change(None)

    def on_kpoint_change(self, c):
        with self.kpnt_out:
            clear_output()
            
            if self.kpoint_slider.value > len(self.selected_cube_files):
                print("Found no cube files")
                self.selected_data = None
                self.height_slider.options = {"---":0}

            else:
                if isinstance(self.selected_cube_files[self.kpoint_slider.value-1], ArrayData):
                    res = self.selected_cube_files[self.kpoint_slider.value-1]
                    self.selected_data = res.get_array('data')
                    cell = res.get_array('voxel') / ANG_2_BOHR
                    for i in range(3):
                        cell[i,:] *= self.selected_data.shape[i]
                    self.selected_atoms = ase.Atoms(
                        numbers=res.get_array('atomic_numbers'),
                        positions=res.get_array('coordinates')/ANG_2_BOHR,
                        cell=cell,
                    )
                else: 
                    absfn = self.selected_cube_files[self.kpoint_slider.value-1]
                    self.selected_data, self.selected_atoms = ase.io.cube.read_cube_data(absfn)
                nz = self.selected_data.shape[2]
                dz = self.selected_atoms.cell[2][2] / nz
                zmid = self.selected_atoms.cell[2][2] / 2.0
                options = OrderedDict()

                for i in range(0,nz,3):
                    z = dz*i
                    options[u"{:.3f} Å".format(z)] = i                    
                self.height_slider.options = options
                nopt = int(len(options)/2)

                self.height_slider.value = list(options.values())[0]

                # Plot 2d
                self.on_orb_plot_change(None) 

                # Plot 3d
                self.orbital_3d()

                self.on_orb_plot_change(None) 

    def on_orb_plot_change(self, c):
        with self.orb_out:
            clear_output()
            if self.selected_data is None:
                return
            fig, ax = plt.subplots()
            fig.dpi = 150.0
            vmin = self.colormap_slider.value/100.0
            vmax = self.colormap_slider.value            

            cax = plot_cube(ax, self.selected_data, self.selected_atoms, self.height_slider.value, 'seismic',vmin,vmax)
            fig.colorbar(cax, label='e/bohr^3', ticks=[vmin, vmax], format='%.0e', orientation='horizontal', shrink=0.3)

            self.plot_overlay_struct(ax, self.selected_atoms, self.orb_alpha_slider.value)
            plt.show()
                   
    def plot_overlay_struct(self, ax,atoms, alpha):
        if alpha == 0:
            return

        # plot overlayed structure
        s = atoms.repeat((2,1,1))
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
    @property
    def spindensity_calc(self):
        try:
            return get_calc_by_label(self._workcalc, "export_spinden")
        except:
            return None

    def spindensity_2d(self):
        if self.spindensity_calc:
            try:
                res = self.spindensity_calc.outputs.output_data
                data = res.get_array('data')
                cell = res.get_array('voxel') / ANG_2_BOHR
                for i in range(3):
                    cell[i,:] *= data.shape[i]
                atoms = ase.Atoms(
                    numbers=res.get_array('atomic_numbers'),
                    positions=res.get_array('coordinates')/ANG_2_BOHR,
                    cell=cell,
                )
            except exceptions.NotExistent:
                data, atoms = ase.io.cube.read_cube_data(self.spindensity_calc.outputs.retrieved.open("_spin.cube.gz").name)
            
            datascaled = data*2000
            def on_spinden_plot_change(c):
                with spinden_out:
                    clear_output()
                    fig, ax = plt.subplots()
                    fig.dpi = 150.0
                    cax = plot_cube(ax, datascaled, atoms, 1, 'seismic')
                    fig.colorbar(cax,  label='arbitrary unit')
                    self.plot_overlay_struct(ax, atoms, spinden_alpha_slider.value)
                    plt.show()

            spinden_alpha_slider = ipw.FloatSlider(description="opacity", value=0.5, max=1.0, continuous_update=False)
            spinden_alpha_slider.observe(on_spinden_plot_change, names='value')
            spinden_out = ipw.Output()
            on_spinden_plot_change(None)
            return ipw.VBox([spinden_out, spinden_alpha_slider])
        else:
            print("Could not find spin density")
            return ipw.HTML('')

    def spindensity_3d(self):
        if self.spindensity_calc:
            file_path= None
            retrieved_files = self.spindensity_calc.outputs.retrieved.list_object_names()
            if "_spin.cube.gz" in retrieved_files:
                file_path = self.spindensity_calc.outputs.retrieved.open("_spin.cube.gz").name
                data, atoms = ase.io.cube.read_cube_data(file_path)
            else:
                try:
                    res = self.spindensity_calc.outputs.output_data
                    data = res.get_array('data')
                    cell = res.get_array('voxel') / ANG_2_BOHR
                    for i in range(3):
                        cell[i,:] *= data.shape[i]
                    atoms = ase.Atoms(
                        numbers=res.get_array('atomic_numbers'),
                        positions=res.get_array('coordinates')/ANG_2_BOHR,
                        cell=cell,
                    )
                except exceptions.NotExistent:
                    print("Spin density cube could not be visualized, file was not retrieved.")
                    return ipw.HTML('')

            ngl_view = nglview.NGLWidget()
            setup_cube_plot(data, atoms, ngl_view)
            isosurf_slider = ipw.FloatSlider(continuous_update=False,
                value=1e-3,
                min=1e-4,
                max=1e-2,
                step=1e-4, 
                description='isovalue',
                readout_format='.1e'
            )
            isosurf_slider.observe(lambda c: set_cube_isosurf([c['new'], -c['new']], ['red', 'blue'], ngl_view), names='value')
            set_cube_isosurf([isosurf_slider.value, -isosurf_slider.value], ['red', 'blue'], ngl_view)
            return ipw.VBox([ngl_view, isosurf_slider])
        else:
            return ipw.HTML('')
            
    def orbital_3d(self):
        if self.selected_data is None:
            return

        # delete all old components
        while hasattr(self.orbital_ngl, "component_0"):
            self.orbital_ngl.component_0.clear_representations()
            cid = self.orbital_ngl.component_0.id
            self.orbital_ngl.remove_component(cid)

        setup_cube_plot(self.selected_data, self.selected_atoms, self.orbital_ngl)
        set_cube_isosurf([self.orb_isosurf_slider.value], ['red'], self.orbital_ngl)
