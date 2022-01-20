# -*- coding: utf-8 -*-
"""Viewers to display the results of the Nanoribbon work chain."""

# Base imports.
import gzip
import re
import io
import tempfile
from base64 import b64encode
from collections import OrderedDict
import nglview
import ipywidgets as ipw
import bqplot as bq
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from IPython.display import display, clear_output
import scipy.constants as const
import ase
import ase.io.cube
from traitlets import dlink, observe, Instance, Int

import copy

# AiiDA imports.
from aiida.common import exceptions
from aiida.plugins import DataFactory

# Local imports.
from .utils import plot_struct_2d, get_calc_by_label, get_calcs_by_label, from_cube_to_arraydata
from .igor import Wave2d

# AiiDA data objects.
ArrayData = DataFactory("array")  # pylint: disable=invalid-name
BandsData = DataFactory("array.bands")  # pylint: disable=invalid-name
StructureData = DataFactory("structure")  # pylint: disable=invalid-name

ANG_2_BOHR = 1.889725989

NANORIBBON_INFO = """
WorkCalculation PK: {pk} <br>
Total energy (eV): {energy} <br>
Band gap (eV): {gap} <br>
Total magnetization/A: {totmagn} <br>
Absolute magnetization/A: {absmagn}
"""


class BandsViewerWidget(ipw.VBox):
    """Widget to view AiiDA BandsData object."""
    bands = Instance(BandsData, allow_none=True)
    structure = Instance(StructureData, allow_none=True)
    selected_band = Int(allow_none=True)
    selected_kpoint = Int(allow_none=True)
    selected_spin = Int(allow_none=True)
    selected_3D = Int(allow_none=True)
    
    def __init__(self, **kwargs):
        self.bands = kwargs['bands']
        self.structure = kwargs['structure']
        self.vacuum_level = kwargs['vacuum_level']
        self.bands_array = self.bands.get_bands()
        self.band_plots = []
        self.homo = kwargs['homo']
        self.lumo = kwargs['lumo']
        
        self.num_export_bands = kwargs['num_export_bands']

        # Always make the array 3-dimensional.
        nptk_ks = 12 #this is hardcoded in bands_lowres for the moment
        if self.bands_array.ndim == 2:
            self.bands_array = self.bands_array[None, :, :]
        self.eff_mass_parabolas = []

        layout = ipw.Layout(padding="5px", margin="0px")
        layout = ipw.Layout(padding="5px", margin="0px", width='auto')

        # Slider to control how many points of the band to use for parabolic fit.
        self.efm_fit_slider = ipw.IntSlider(description="Eff. mass fit",
                                            min=3,
                                            max=15,
                                            step=2,
                                            continuous_update=False,
                                            layout=layout)
        band_selector = ipw.IntSlider(description="Band",
                                      value=int(kwargs['nelectrons'] / 2) ,
                                      min=max(1,int(kwargs['nelectrons'] / 2) - self.num_export_bands//2),
                                      max=int(kwargs['nelectrons'] / 2) + self.num_export_bands//2,
                                      step=1,
                                      continuous_update=False,
                                      readout_format='d',
                                      layout=layout)
        kpoint_slider = ipw.IntSlider(description="k-point",
                                      min=1,
                                      max=nptk_ks,
                                      readout_format='d',
                                      continuous_update=False,
                                      layout=layout)
        spin_selector = ipw.RadioButtons(options=[('up', 0), ('down', 1)],
                                              description='Select spin',
                                              disabled=False)
        view_3D = ipw.RadioButtons(options=[('no', 0), ('yes', 1)],
                                              description='plot3D',
                                              disabled=False)        

        boxes = [self.efm_fit_slider, band_selector, kpoint_slider, spin_selector,view_3D]

        plots = []
        for ispin in range(self.bands_array.shape[0]):
            box, plot, eff_mass_parabola = self.plot_bands(ispin)
            plots.append(box)
            self.band_plots.append(plot)
            self.eff_mass_parabolas.append(eff_mass_parabola)
        boxes.append(ipw.HBox(plots))

        dlink((kpoint_slider, 'value'), (self, 'selected_kpoint'))
        dlink((band_selector, 'value'), (self, 'selected_band'))
        dlink((spin_selector, 'value'), (self, 'selected_spin'))
        dlink((view_3D, 'value'), (self, 'selected_3D'))

        # Display the orbital map also initially.
        self.on_band_change(_=None)

        super().__init__(boxes, **kwargs)

    def plot_bands(self, ispin):
        """Plot band structure."""
        _, nkpoints, nbands = self.bands_array.shape
        center = (self.homo + self.lumo) / 2.0
        x_sc = bq.LinearScale()
        y_sc = bq.LinearScale(
            min=center - 3.0,
            max=center + 3.0,
        )

        x_max = np.pi / self.structure.cell_lengths[0]

        x_data = np.linspace(0.0, x_max, nkpoints)
        y_datas = self.bands_array[ispin, :, :].transpose() - self.vacuum_level

        lines = bq.Lines(x=x_data,
                         y=y_datas,
                         color=np.zeros(nbands),
                         animate=True,
                         stroke_width=4.0,
                         scales={
                             'x': x_sc,
                             'y': y_sc,
                             'color': bq.ColorScale(colors=['gray', 'red'], min=0.0, max=1.0)
                         })

        homo_line = bq.Lines(x=[0, x_max],
                             y=[center, center],
                             line_style='dashed',
                             colors=['red'],
                             scales={
                                 'x': x_sc,
                                 'y': y_sc
                             })

        # Initialize the parabola as a random line and set visible to false
        # Later, when it is correctly set, show it.
        eff_mass_parabola = bq.Lines(x=[0, 0],
                                     y=[0, 0],
                                     visible=False,
                                     stroke_width=1.0,
                                     line_style='solid',
                                     colors=['blue'],
                                     scales={
                                         'x': x_sc,
                                         'y': y_sc
                                     })
        ax_x = bq.Axis(label=u'kA^-1', scale=x_sc, grid_lines='solid', tick_format='.3f', tick_values=[0, x_max])
        ax_y = bq.Axis(label='eV', scale=y_sc, orientation='vertical', grid_lines='solid')

        fig = bq.Figure(axes=[ax_x, ax_y],
                        marks=[lines, homo_line, eff_mass_parabola],
                        title='Spin {}'.format(ispin),
                        layout=ipw.Layout(height="800px", width="200px"),
                        fig_margin={
                            "left": 45,
                            "top": 60,
                            "bottom": 60,
                            "right": 40
                        },
                        min_aspect_ratio=0.25,
                        max_aspect_ratio=0.25)

        save_btn = ipw.Button(description="Download png")
        save_btn.on_click(lambda b: fig.save_png())  # save_png() does not work with unicode labels

        box = ipw.VBox([fig, save_btn, self.mk_igor_link(ispin)],
                       layout=ipw.Layout(align_items="center", padding="5px", margin="0px"))
        return box, lines, eff_mass_parabola

    def mk_igor_link(self, ispin):
        """Create a downloadable link."""
        igorvalue = self.igor_bands(ispin)
        igorfile = b64encode(igorvalue.encode()).decode()
        filename = self.structure.get_ase().get_chemical_formula() + "_bands_spin{}_pk{}.itx".format(
            ispin, self.structure.id)

        html = '<a download="{}" href="'.format(filename)
        html += 'data:chemical/x-igor;name={};base64,{}"'.format(filename, igorfile)
        html += ' id="bands_link"'
        html += ' target="_blank">Export itx-Bands</a>'
        return ipw.HTML(html)

    def igor_bands(self, ispin):
        """Export the band structure in IGOR data format."""
        _, nkpoints, nbands = self.bands_array.shape
        k_axis = np.linspace(0.0, np.pi / self.structure.cell_lengths[0], nkpoints)
        testio = io.StringIO()
        tosave = self.bands_array[ispin, :, :].transpose() - self.vacuum_level

        with testio as fobj:
            fobj.write(u'IGOR\r')
            fobj.write(u'WAVES')
            fobj.write(u'\tx1' + (u'\ty{}' * nbands).format(*[x for x in range(nbands)]) + u'\r')
            fobj.write(u'BEGIN\r')
            for i in range(nkpoints):
                fobj.write(u"\t{:.7f}".format(k_axis[i]))  # first column k_axis
                fobj.write((u"\t{:.7f}" * nbands).format(*tosave[:, i]))  # other columns the bands
                fobj.write(u"\r")
            fobj.write(u"END\r")
            fobj.write(u'X SetScale/P x {},{},"", x1; SetScale y 0,0,"", x1\r'.format(0, k_axis[1] - k_axis[0]))
            for idk in range(nbands):
                fobj.write((u'X SetScale/P x 0,1,"", y{0}; SetScale y 0,0,"", y{0}\r').format(str(idk)))
            return testio.getvalue()

    @observe('selected_band')
    def on_band_change(self, _=None):
        """Highlight the selected band."""
        #self.selected_spin = self.spin_selector.value
        nspins, _, nbands = self.bands_array.shape

        colors = np.zeros((nspins, nbands))
        colors[self.selected_spin*(nspins-1), self.selected_band -1] = 1.0

        for ispin in range(nspins):
            self.band_plots[ispin].color = colors[ispin, :]
          
    @observe('selected_spin')  
    def on_spin_change(self, _=None):
        """Highlight the selected spin channel."""    
        nspins, _, nbands = self.bands_array.shape

        colors = np.zeros((nspins, nbands))
        colors[self.selected_spin*(nspins-1), self.selected_band -1] = 1.0

        for ispin in range(nspins):
            self.band_plots[ispin].color = colors[ispin, :]        

    def calc_effective_mass(self, ispin):
        """Compute effective mass."""
        # m* = hbar^2*[d^2E/dk^2]^-1
        hbar = const.value('Planck constant over 2 pi in eV s')
        el_mass = const.m_e * 1e-20 / const.eV  # in eV*s^2/ang^2
        _, nkpoints, _ = self.bands_array.shape
        band = self.bands_array[ispin].transpose()[self.selected_band -1] - self.vacuum_level
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
        i_min = parabola_ind - int(np.ceil(num_fit_points / 2.0)) + len(band)
        i_max = parabola_ind + int(np.floor(num_fit_points / 2.0)) + len(band)

        fit_energies = band_ext[i_min:i_max]
        fit_kvals = k_vals_ext[i_min:i_max]

        parabola_fit = np.polyfit(fit_kvals, fit_energies, 2)

        meff = hbar**2 / (2 * parabola_fit[0]) / el_mass

        # restrict fitting values to "main region"
        main_region_mask = (fit_kvals >= k_axis[0]) & (fit_kvals <= k_axis[-1])
        fit_energies = fit_energies[main_region_mask]
        fit_kvals = fit_kvals[main_region_mask]

        return meff, parabola_fit, fit_kvals, fit_energies


class CubeArrayData3dViewerWidget(ipw.VBox):
    """Widget to View 3-dimensional AiiDA ArrayData object in 3D."""

    arraydata = Instance(ArrayData, allow_none=True)

    def __init__(self, **kwargs):
        self.data_3d = None
        self.units = None
        self.structure = None
        self.viewer = nglview.NGLWidget()
        self.orb_isosurf_slider = ipw.FloatSlider(continuous_update=False,
                                                  value=1e-3,
                                                  min=1e-4,
                                                  max=1e-2,
                                                  step=1e-4,
                                                  description='Isovalue',
                                                  readout_format='.1e')
        self.orb_isosurf_slider.observe(
            lambda c: self.set_cube_isosurf(  # pylint: disable=no-member
                [c['new'], -c['new']], ['red', 'blue']),
            names='value')
        super().__init__([self.viewer, self.orb_isosurf_slider], **kwargs)

    @observe('arraydata')
    def on_observe_arraydata(self, _=None):
        """Update object attributes when arraydata trait is modified."""

        self.data_3d = self.arraydata.get_array('data')
        cell = self.arraydata.get_array('voxel') / ANG_2_BOHR
        for i in range(3):
            cell[i, :] *= self.data_3d.shape[i]
        self.structure = ase.Atoms(
            numbers=self.arraydata.get_array('atomic_numbers'),
            positions=self.arraydata.get_array('coordinates') / ANG_2_BOHR,
            cell=cell,
            pbc=True,
        )
        self.units = self.arraydata.get_array('data_units')
        self.update_plot()

    def update_plot(self):
        """Update the 3D plot."""
        # pylint: disable=no-member
        while hasattr(self.viewer, "component_0"):
            self.viewer.component_0.clear_representations()
            self.viewer.remove_component(self.viewer.component_0.id)
        self.setup_cube_plot()
        self.set_cube_isosurf(
            [self.orb_isosurf_slider.value, -self.orb_isosurf_slider.value],  # pylint: disable=invalid-unary-operand-type
            ['red', 'blue'])

    def setup_cube_plot(self):
        """Setup cube plot."""
        # pylint: disable=no-member
        n_repeat = 2
        atoms_xn = self.structure.repeat((n_repeat, 1, 1))
        data_xn = np.tile(self.data_3d, (n_repeat, 1, 1))
        self.viewer.add_component(nglview.ASEStructure(atoms_xn))
        with tempfile.NamedTemporaryFile(mode='w') as tempf:
            ase.io.cube.write_cube(tempf, atoms_xn, data_xn)
            c_2 = self.viewer.add_component(tempf.name, ext='cube')
            c_2.clear()

    def set_cube_isosurf(self, isovals, colors):
        """Set cube isosurface."""
        # pylint: disable=no-member
        if hasattr(self.viewer, 'component_1'):
            c_2 = self.viewer.component_1
            c_2.clear()
            for isov, col in zip(isovals, colors):
                c_2.add_surface(color=col, isolevelType="value", isolevel=isov)


class CubeArrayData2dViewerWidget(ipw.VBox):
    """Widget to View 3-dimensional AiiDA ArrayData object projected on 2D plane."""
    arraydata = Instance(ArrayData, allow_none=True)

    def __init__(self, cmap='seismic', center0=True, show_cbar=True, export_label=None, **kwargs):
                
        self.cmap = cmap
        self.center0 = center0
        self.show_cbar = show_cbar
        self.export_label = export_label
        
        self.structure = None
        self._current_structure = None
        self.selected_data = None
        self._current_data = None
        self.units = None
        layout = ipw.Layout(padding="5px", margin="0px")
        self.plot = ipw.Output(layout=layout)
        self.height_slider = ipw.SelectionSlider(description="Height",
                                                 options={"---": 0},
                                                 continuous_update=False,
                                                 layout=layout)
        self.height_slider.observe(self.update_plot, names='value')
        self.selected_height = 0.0 # angstrom
        
        self.opacity_slider = ipw.FloatSlider(description="Opacity",
                                              value=0.5,
                                              max=1.0,
                                              continuous_update=False,
                                              layout=layout)
        self.opacity_slider.observe(self.update_plot, names='value')
        self.colormap_slider = ipw.FloatSlider(value=1.0,
                                                  min=0.05,
                                                  max=5.0,
                                                  step=0.05,
                                                  description='Color max.',
                                                  continuous_update=False,
                                                  layout=layout)
        self.colormap_slider.observe(self.update_plot, names='value')

        self.axis = ipw.ToggleButtons(options=[('X', [1, 2, 0]), ('Y', [2, 0, 1]), ('Z', [0, 1, 2])],
                                      value=[0, 1, 2],
                                      description='Axis:',
                                      orientation='horizontal',
                                      disabled=True,
                                      style={'button_width': 'initial'})
        self.axis.observe(self.update_axis, names='value')
        
        self.dl_link = ipw.HTML(value="")

        super().__init__([self.plot, ipw.HBox([self.axis, self.dl_link]),
                          self.height_slider,
                          self.colormap_slider, self.opacity_slider],
                         **kwargs)

    @observe('arraydata')
    def on_observe_arraydata(self, _=None):
        """Update object attributes when arraydata trait is modified."""
        self.selected_data = self.arraydata.get_array('data')
        cell = self.arraydata.get_array('voxel') / ANG_2_BOHR
        for i in range(3):
            cell[i, :] *= self.selected_data.shape[i]
        self.structure = ase.Atoms(
            numbers=self.arraydata.get_array('atomic_numbers'),
            positions=self.arraydata.get_array('coordinates') / ANG_2_BOHR,
            cell=cell,
            pbc=True,
        )
        self.units = self.arraydata.get_array('data_units')
        self.update_axis()
        self.update_plot()

    def update_axis(self, _=None):
        """Make plot perpencicular to the selected axis."""
        # Adapting (transposing) the data
        self._current_data = self.selected_data.transpose(self.axis.value)

        # Adapting (transposing) coordinates and positions.
        rot = {
            'X': ('x', 'z'),
            'Y': ('y', 'z'),
            'Z': ('z', 'z'),
        }
        self._current_structure = self.structure.copy()
        self._current_structure.rotate(*rot[self.axis.label], rotate_cell=True)

        n_z = self._current_data.shape[2]
        d_z = self._current_structure.cell[2][2] / n_z
        options = OrderedDict()
        
        geo_center = np.sum(self._current_structure.positions, axis=0) / len(self._current_structure)
        
        self.z_arr = np.array([d_z * i - geo_center[2] for i in range(0, n_z, 1)])

        for i, z in enumerate(self.z_arr):
            options[u"{:.3f} Å".format(z)] = i
            
        self.height_slider.options = options
        
        i_start = (np.abs(self.z_arr - 2.5)).argmin()
        self.height_slider.value = list(options.values())[i_start]
        self.selected_height = self.z_arr[i_start]

    def update_plot(self, _=None):
        """Update the 2D plot with the new data."""
        with self.plot:
            clear_output()
            fig, axplt = plt.subplots()
            fig.dpi = 150.0
            data = self._current_data[:, :, self.height_slider.value]
            self.selected_height = self.z_arr[self.height_slider.value]
            flipped_data = np.flip(data.transpose(), axis=0)
            
            vmax = np.max(flipped_data) * self.colormap_slider.value
            vmin = np.min(flipped_data) * self.colormap_slider.value
            
            if vmax < 0: vmax = 0.0
            if vmin > 0: vmin = 0.0

            x_2 = self._current_structure.cell[0][0] * 2.0
            y_2 = self._current_structure.cell[1][1]

            # Set labels and limits.
            axplt.set_xlabel(u'Å')
            axplt.set_ylabel(u'Å')
            axplt.set_xlim(0, x_2)
            axplt.set_ylim(0, y_2)
            
            if self.center0:
                amax = np.max(np.abs([vmax, vmin]))
                vmin = -amax
                vmax = amax
                
            plot = axplt.imshow(np.tile(flipped_data, (1, 2)), extent=[0, x_2, 0, y_2],
                         cmap=self.cmap, vmin=vmin, vmax=vmax)
            
            if self.show_cbar:
                fig.colorbar(plot,
                             label=self.units,
                             ticks=[vmin, vmax],
                             format='%.e',
                             orientation='horizontal',
                             shrink=0.3)

            # To show structure uniformly with transparency, add fully opaque structure and
            # then an additional transparent data layer on top
            plot_struct_2d(axplt, self._current_structure, 1.0)
            axplt.imshow(np.tile(flipped_data, (1, 2)), extent=[0, x_2, 0, y_2],
                         cmap=self.cmap, vmin=vmin, vmax=vmax, alpha=(1-self.opacity_slider.value), zorder=10)
            
            plt.show()
            
            self.make_export_links(data)
    
    def make_export_links(self, data):
        html = "&nbsp;&nbsp;&nbsp;&nbsp;Download: "
        html += self.make_export_link_txt(data)
        html += " "
        html += self.make_export_link_igor(data)
        
        self.dl_link.value = html
        
    def make_export_link_txt(self, data):
        header = "xlim=(%.2f, %.2f), ylim=(%.2f, %.2f)" % (0.0, self._current_structure.cell[0][0],
                                                           0.0, self._current_structure.cell[1][1])
        tempio = io.BytesIO()
        np.savetxt(tempio, data, header=header, fmt="%.4e")
        
        enc_file = b64encode(tempio.getvalue()).decode()
        
        if self.export_label is not None:
            plot_name = self.export_label
        else:
            plot_name = "export"
        
        filename = "{}_h{:.3f}.txt".format(plot_name, self.selected_height)

        html = '<a download="{}" href="'.format(filename)
        html += 'data:chemical/txt;name={};base64,{}"'.format(filename, enc_file)
        html += ' id="export_link"'
        html += ' target="_blank">.txt</a>'
        
        return html
        
    def make_export_link_igor(self, data):
        if self.export_label is not None:
            plot_name = self.export_label
        else:
            plot_name = "export"
            
        filename = "{}_h{:.3f}.itx".format(plot_name, self.selected_height)
        
        igorwave = Wave2d(
                data=data,
                xmin=0.0,
                xmax=self._current_structure.cell[0][0],
                xlabel='x [Angstroms]',
                ymin=0.0,
                ymax=self._current_structure.cell[1][1],
                ylabel='y [Angstroms]',
                name="'%s'" % plot_name,
        )
        
        enc_file = b64encode(str(igorwave).encode()).decode()

        html = '<a download="{}" href="'.format(filename)
        html += 'data:chemical/itx;name={};base64,{}"'.format(filename, enc_file)
        html += ' id="export_link"'
        html += ' target="_blank">.itx</a>'
        
        return html

class NanoribbonShowWidget(ipw.VBox):
    """Show the results of a nanoribbon work chain."""

    def __init__(self, workcalc, **kwargs):
        self._workcalc = workcalc
        

        self.info = ipw.HTML(
            NANORIBBON_INFO.format(
                pk=workcalc.id,
                energy=workcalc.extras['total_energy'],
                gap=workcalc.extras['gap'],
                totmagn=workcalc.extras['total_magnetization_per_angstr'],
                absmagn=workcalc.extras['absolute_magnetization_per_angstr'],
            ))

        self.orbitals_calcs = get_calcs_by_label(workcalc, "export_orbitals")
        prev_calc = self.orbitals_calcs[0].inputs.parent_folder.creator
        self.nkpoints_lowres = prev_calc.res.number_of_k_points
        
        self.bands_lowres = prev_calc.outputs.output_band.get_bands() # [spin, kpt, band]
        # In case of RKS calculation, the spin dimension is not present, add it for convenience
        if self.bands_lowres.ndim == 2:
            self.bands_lowres = np.expand_dims(self.bands_lowres, axis=0)
        
        self.vacuum_level = self._workcalc.get_extra('vacuum_level')
        
        self.list_of_calcs = []
        for orbitals_calc in self.orbitals_calcs:
            if 'output_data_multiple' in orbitals_calc.outputs:
                self.list_of_calcs += [(k, v) for k, v in dict(orbitals_calc.outputs.output_data_multiple).items()]
            elif any(['output_data_multiple' in x for x in orbitals_calc.outputs]):
                self.list_of_calcs += [(x, orbitals_calc) for x in orbitals_calc.outputs]
            else:
                self.list_of_calcs += [(x.name, orbitals_calc) for x in orbitals_calc.outputs.retrieved.list_objects()]
                
        # How many bands were exported?
        if "num_export_bands" in workcalc.inputs:
            self.num_export_bands = workcalc.inputs.num_export_bands
        else:
            self.num_export_bands = 2 # in old versions it was hardcoded as 2
        
        bands_calc = get_calc_by_label(workcalc, "bands")
        self.nspin = bands_calc.outputs.output_band.get_bands().ndim -1
        self.selected_cube_files = []
        self.bands_viewer = BandsViewerWidget(
            bands=bands_calc.outputs.output_band,
            nelectrons=int(bands_calc.outputs.output_parameters['number_of_electrons']),
            vacuum_level=self.vacuum_level,
            structure=bands_calc.inputs.structure,
            homo=self._workcalc.get_extra('homo'),
            lumo=self._workcalc.get_extra('lumo'),
            gap=self._workcalc.get_extra('gap'),
            num_export_bands=self.num_export_bands,
        )
        self.bands_viewer.observe(self.on_kpoint_change, names=['selected_band', 'selected_kpoint', 'selected_spin'])
        self.bands_viewer.observe(self.on_view_3D_change, names='selected_3D')
        
        # Custom cmap for orbital 2d viewer
        custom_cmap_colors = [
            [0.00, (1.0, 1.0, 1.0)],
            [0.50, (1.0, 0.0, 0.0)],
            [1.00, (0.5, 0.0, 0.0)],
        ]
        custom_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("half_seismic", custom_cmap_colors)
        
        self.orbital_viewer_2d = CubeArrayData2dViewerWidget(
            cmap=custom_cmap,
            center0=False,
            export_label=None
        )
        self.orbital_viewer_3d = CubeArrayData3dViewerWidget()
        
        self.info_out = ipw.HTML()
        
        spin_density_vbox = ipw.VBox([])
        
        if self.spindensity_calc:
            
            self.spinden_viewer_2d = CubeArrayData2dViewerWidget(cmap='seismic', export_label="pk%d_spin" % workcalc.pk)
            self.spinden_viewer_3d = CubeArrayData3dViewerWidget()

            self.output_s = ipw.Output()

            self.spin_view_3D = ipw.RadioButtons(options=[('no', 0), ('yes', 1)],
                                          description='plot spin 3D',
                                          disabled=False)  

            self.spin_view_3D.observe(self.on_spin_view_mode_change, names='value')
            
            spin_density_vbox.children += tuple([ipw.HTML(value="<h1>Spin density</h1>")])
            spin_density_vbox.children += tuple([self.spin_view_3D])
            spin_density_vbox.children += tuple([self.output_s])
            
            self.on_spin_view_mode_change()
        
        # ---------------------------------------
        # STS mapping widget
        self.sts_heading = ipw.HTML(value="<h1>LDOS mappings</h1>")
        self.sts_energy_text = ipw.FloatText(value=round(self._workcalc.get_extra('homo'), 2), description='energy [eV]:')
        self.sts_fwhm_text = ipw.FloatText(value=0.1, description='fwhm [eV]:')
        self.sts_mapping_viewer = CubeArrayData2dViewerWidget(cmap='gist_heat', center0=False,
                                                              show_cbar=False, export_label="pk%d_ldos" % workcalc.pk)
        self.sts_mapping_viewer_wrapper = ipw.VBox([])
        self.sts_btn = ipw.Button(description='view LDOS')
        self.sts_btn.on_click(self.on_sts_btn_click)
        
        self.sts_viewer_box = ipw.VBox([
            self.sts_heading,
            ipw.HBox([self.sts_energy_text, self.sts_fwhm_text, self.sts_btn]),
            self.sts_mapping_viewer_wrapper
        ])
        # ---------------------------------------
            
        self.output = ipw.Output()

        self.on_kpoint_change()
        
        super().__init__([
            self.info,
            ipw.HBox([self.bands_viewer, self.output]), spin_density_vbox, self.sts_viewer_box
        ], **kwargs)

    
    def on_spin_view_mode_change(self, _=None):
        
        def _set_viewer_cube_data(viewer):
            try:
                viewer.arraydata = self.spindensity_calc.outputs.output_data
            except exceptions.NotExistent:
                print("Compatibility mode for old spin cube file")
                with gzip.open(self.spindensity_calc.outputs.retrieved.open("_spin.cube.gz").name) as fpointer:
                    arrayd = from_cube_to_arraydata(fpointer.read())
                    viewer.arraydata = arrayd
        
        with self.output_s:
            clear_output()
            if self.spin_view_3D.value == 0:
                display(self.spinden_viewer_2d)
                _set_viewer_cube_data(self.spinden_viewer_2d)
            else:
                display(self.spinden_viewer_3d)
                _set_viewer_cube_data(self.spinden_viewer_3d)
        
    
    def on_view_3D_change(self, _=None):
        """Plot 3D orbitals in case of selection."""
        if self.bands_viewer.selected_3D:
            self.twod_3D=[self.orbital_viewer_2d, self.orbital_viewer_3d]
        else:    
            self.twod_3D=[self.orbital_viewer_2d]
        self.on_kpoint_change(None)
    
    
    def _read_arraydata(self, i_spin, i_kpt, i_band):
        
        spin_mult = self.nspin - 1
        kpt_qe_convention = i_kpt + 1 + self.nkpoints_lowres * i_spin*spin_mult
        
        cube_id = [i for i, f in enumerate(self.list_of_calcs)
                   if 'K'+str(kpt_qe_convention).zfill(3)+'_'+'B'+str(i_band+1).zfill(3) in list(f)[0]]
        
        if len(cube_id)==0:
            return None
        
        cid=cube_id[0]
        fname = list(self.list_of_calcs[cid])[0]
        
        if fname[0] == 'K' and fname[4:6] == '_B':
            arraydata = list(self.list_of_calcs[cid])[1]
        elif fname.startswith('output_data_multiple'):
            arraydata = list(self.list_of_calcs[cid])[1].outputs[fname]
        else:
            absfn = list(self.list_of_calcs[cid])[1].outputs.retrieved.open(fname).name
            with gzip.open(absfn) as fpointer:
                arraydata = from_cube_to_arraydata(fpointer.read())
                
        return arraydata, fname
    
    def clamp_arraydata_to_zero(self, arraydata):
        new_ad = copy.deepcopy(arraydata)
        data = new_ad.get_array('data')
        new_ad.set_array('data', np.maximum(data, 0.0))
        return new_ad
        
    def on_kpoint_change(self, _=None):
        """Replot the orbitals in case of the kpoint change."""
        
        arraydata_fn = self._read_arraydata(self.bands_viewer.selected_spin,
                                         self.bands_viewer.selected_kpoint-1,
                                         self.bands_viewer.selected_band-1)
        if arraydata_fn is None:
            with self.output:
                clear_output()
                self.info_out.value = "Found no cube files"
                #self.orbital_viewer_3d.arraydata = None
                #self.orbital_viewer_2d.arraydata = None
                display(ipw.VBox([self.info_out], layout=ipw.Layout(margin="200px 0px 0px 0px")))
        else:
            arraydata, fname = arraydata_fn
            self.info_out.value = fname
            
            self.orbital_viewer_2d.export_label = "pk{}_b{}_k{:02}_s{}".format(
                self._workcalc.pk,
                self.bands_viewer.selected_band,
                self.bands_viewer.selected_kpoint,
                self.bands_viewer.selected_spin,
            )
            self.orbital_viewer_2d.arraydata = self.clamp_arraydata_to_zero(arraydata)
            
            with self.output:
                clear_output()                        
                if self.bands_viewer.selected_3D:
                    self.orbital_viewer_3d.arraydata = arraydata
                    hbox = ipw.HBox([self.orbital_viewer_3d])
                else:
                    hbox = ipw.HBox([self.orbital_viewer_2d])
                display(ipw.VBox(
                    [self.info_out, hbox],
                    layout=ipw.Layout(margin="200px 0px 0px 0px")
                ))
    
    def gaussian(self, x, fwhm):
        sigma = fwhm/2.3548
        return np.exp(-x**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))
    
    def calculate_sts_mapping(self, energy, broadening):
        
        sts_arraydata = None
        sts_arraydata_meta = None
        warn = False
        
        for i_band in range(self.bands_lowres.shape[2]-1, -1, -1):
            for i_spin in range(self.bands_lowres.shape[0]):
                for i_kpt in range(self.bands_lowres.shape[1]):
                    bz_w = 1 if i_kpt in [0, 12] else 2
                    
                    orb_energy = self.bands_lowres[i_spin, i_kpt, i_band] - self.vacuum_level
                    
                    if np.abs(energy - orb_energy) < 2 * broadening:
                        
                        coef = self.gaussian(energy - orb_energy, broadening)
                        
                        ad_out = self._read_arraydata(i_spin, i_kpt, i_band)
                        
                        if ad_out is None:
                            warn = True
                            continue
                            
                        arraydata, fname = ad_out
                        arraydata = self.clamp_arraydata_to_zero(arraydata)
                        
                        orbital_data = arraydata.get_array('data')
                        
                        if sts_arraydata is None:
                            sts_arraydata = bz_w * coef * orbital_data
                            sts_arraydata_meta = copy.deepcopy(arraydata)
                        else:
                            sts_arraydata += bz_w * coef * orbital_data
        
        if sts_arraydata_meta is None:
            return None, warn
        else:
            sts_arraydata_meta.set_array('data', sts_arraydata)
            return sts_arraydata_meta, warn
    
    def on_sts_btn_click(self, _=None):
        
        sts_arraydata, warn = self.calculate_sts_mapping(self.sts_energy_text.value, self.sts_fwhm_text.value)
        
        if sts_arraydata is not None:
            if warn:
                self.sts_mapping_viewer_wrapper.children = [
                    ipw.HTML(value="Warning: some relevant bands were not found"),
                    self.sts_mapping_viewer
                ]
            else:
                self.sts_mapping_viewer_wrapper.children = [self.sts_mapping_viewer]
            
            self.sts_mapping_viewer.export_label = "pk{}_ldos_fwhm{}_e{}".format(
                self._workcalc.pk, self.sts_fwhm_text.value, self.sts_energy_text.value
            )
            self.sts_mapping_viewer.arraydata = sts_arraydata
        else:
            self.sts_mapping_viewer_wrapper.children = [ipw.HTML(value="Could not find data.")]
        
    
    @property
    def spindensity_calc(self):
        """Return spindensity plot calculation if present, otherwise return None."""
        try:
            return get_calc_by_label(self._workcalc, "export_spinden")
        except AssertionError:
            return None
