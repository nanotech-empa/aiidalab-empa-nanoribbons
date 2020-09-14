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
from IPython.display import clear_output
import scipy.constants as const
import ase
import ase.io.cube
from traitlets import dlink, observe, Instance, Int

# AiiDA imports.
from aiida.common import exceptions
from aiida.plugins import DataFactory

# Local imports.
from .utils import plot_struct_2d, get_calc_by_label, get_calcs_by_label, from_cube_to_arraydata

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

    def __init__(self, **kwargs):
        self.bands = kwargs['bands']
        self.structure = kwargs['structure']
        self.vacuum_level = kwargs['vacuum_level']
        self.bands_array = self.bands.get_bands()
        self.band_plots = []
        self.homo = kwargs['homo']
        self.lumo = kwargs['lumo']

        # Always make the array 3-dimensional.
        nptk_ks = 12 #this is hardcoded in bands_lowres for the moment
        if self.bands_array.ndim == 2:
            self.bands_array = self.bands_array[None, :, :]
        self.eff_mass_parabolas = []

        layout = ipw.Layout(padding="5px", margin="0px")

        # Slider to control how many points of the band to use for parabolic fit.
        self.efm_fit_slider = ipw.IntSlider(description="Eff. mass fit",
                                            min=3,
                                            max=15,
                                            step=2,
                                            continuous_update=False,
                                            layout=layout)
        band_selector = ipw.IntSlider(description="Band",
                                      value=int(kwargs['nelectrons'] / 2) - 1,
                                      min=0,
                                      max=self.bands_array.shape[2],
                                      step=1,
                                      continuous_update=False,
                                      layout=layout)
        kpoint_slider = ipw.IntSlider(description="k-point",
                                      min=1,
                                      max=nptk_ks,
                                      continuous_update=False,
                                      layout=layout)
        self.spin_selector = ipw.RadioButtons(options=[('up', 0), ('down', 1)],
                                              description='Select spin',
                                              disabled=False)

        boxes = [self.efm_fit_slider, band_selector, kpoint_slider, self.spin_selector]

        plots = []
        for ispin in range(self.bands_array.shape[0]):
            box, plot, eff_mass_parabola = self.plot_bands(ispin)
            plots.append(box)
            self.band_plots.append(plot)
            self.eff_mass_parabolas.append(eff_mass_parabola)
        boxes.append(ipw.HBox(plots))

        dlink((kpoint_slider, 'value'), (self, 'selected_kpoint'))
        dlink((band_selector, 'value'), (self, 'selected_band'))

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
                             y=[self.homo, self.homo],
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
        igorfile = b64encode(igorvalue.encode())
        filename = self.structure.get_ase().get_chemical_formula() + "_bands_spin{}_pk{}.itx".format(
            ispin, self.structure.id)

        html = '<a download="{}" href="'.format(filename)
        html += 'data:chemical/x-igor;name={};base64,{}"'.format(filename, igorfile)
        html += ' id="pdos_link"'
        html += ' target="_blank">Export itx-Bands</a>'
        return ipw.HTML(html)

    def igor_bands(self, ispin):
        """Exprot the band structure in IGOR data format."""
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
        self.selected_spin = self.spin_selector.value
        nspins, _, nbands = self.bands_array.shape

        colors = np.zeros((nspins, nbands))
        colors[self.selected_spin, self.selected_band] = 1.0

        for ispin in range(nspins):
            self.band_plots[ispin].color = colors[ispin, :]

    def calc_effective_mass(self, ispin):
        """Compute effective mass."""
        # m* = hbar^2*[d^2E/dk^2]^-1
        hbar = const.value('Planck constant over 2 pi in eV s')
        el_mass = const.m_e * 1e-20 / const.eV  # in eV*s^2/ang^2
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

    def __init__(self, **kwargs):
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
        self.opacity_slider = ipw.FloatSlider(description="Opacity",
                                              value=0.5,
                                              max=1.0,
                                              continuous_update=False,
                                              layout=layout)
        self.opacity_slider.observe(self.update_plot, names='value')
        self.colormap_slider = ipw.FloatLogSlider(value=0.2,
                                                  min=-5,
                                                  max=1,
                                                  step=0.1,
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

        super().__init__([self.plot, self.axis, self.height_slider, self.colormap_slider, self.opacity_slider],
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

        for i in range(0, n_z, 3):
            options[u"{:.3f} Å".format(d_z * i)] = i
        self.height_slider.options = options
        nopt = int(len(options) / 2)
        self.height_slider.value = list(options.values())[nopt]

    def update_plot(self, _=None):
        """Update the 2D plot with the new data."""
        with self.plot:
            clear_output()
            fig, axplt = plt.subplots()
            fig.dpi = 150.0
            vmax = np.max(np.abs(self._current_data)) * self.colormap_slider.value

            flipped_data = np.flip(self._current_data[:, :, self.height_slider.value].transpose(), axis=0)

            x_2 = self._current_structure.cell[0][0] * 2.0
            y_2 = self._current_structure.cell[1][1]

            # Set labels and limits.
            axplt.set_xlabel(u'Å')
            axplt.set_ylabel(u'Å')
            axplt.set_xlim(0, x_2)
            axplt.set_ylim(0, y_2)

            fig.colorbar(axplt.imshow(np.tile(flipped_data, (1, 2)),
                                      extent=[0, x_2, 0, y_2],
                                      cmap='seismic',
                                      vmin=-vmax,
                                      vmax=vmax),
                         label=self.units,
                         ticks=[-vmax, vmax],
                         format='%.e',
                         orientation='horizontal',
                         shrink=0.3)

            plot_struct_2d(axplt, self._current_structure, self.opacity_slider.value)
            plt.show()


class NanoribbonShowWidget(ipw.VBox):
    """Show the results of a nanoribbon work chain."""

    def __init__(self, workcalc, **kwargs):
        self._workcalc = workcalc

        self.info = ipw.HTML(
            NANORIBBON_INFO.format(
                pk=workcalc.id,
                energy=workcalc.extras['total_energy'],
                gap=workcalc.extras['gap'],
                totmagn=workcalc.extras['absolute_magnetization_per_angstr'],
                absmagn=workcalc.extras['total_magnetization_per_angstr'],
            ))

        self.orbitals_calcs = get_calcs_by_label(workcalc, "export_orbitals")
        bands_calc = get_calc_by_label(workcalc, "bands")
        self.selected_cube_files = []
        self.bands_viewer = BandsViewerWidget(
            bands=bands_calc.outputs.output_band,
            nelectrons=int(bands_calc.outputs.output_parameters['number_of_electrons']),
            vacuum_level=self._workcalc.get_extra('vacuum_level'),
            structure=bands_calc.inputs.structure,
            homo=self._workcalc.get_extra('homo'),
            lumo=self._workcalc.get_extra('lumo'),
            gap=self._workcalc.get_extra('gap'),
        )
        self.bands_viewer.observe(self.on_band_change, names='selected_band')
        self.bands_viewer.observe(self.on_kpoint_change, names='selected_kpoint')

        self.orbital_viewer_2d = CubeArrayData2dViewerWidget()
        self.orbital_viewer_3d = CubeArrayData3dViewerWidget()
        self.spinden_viewer_2d = CubeArrayData2dViewerWidget()
        self.spinden_viewer_3d = CubeArrayData3dViewerWidget()
        self.info_out = ipw.HTML()

        if self.spindensity_calc:
            try:
                self.spinden_viewer_2d.arraydata = self.spindensity_calc.outputs.output_data
                self.spinden_viewer_3d.arraydata = self.spindensity_calc.outputs.output_data
            except exceptions.NotExistent:
                with gzip.open(self.spindensity_calc.outputs.retrieved.open("_spin.cube.gz").name) as fpointer:
                    arrayd = from_cube_to_arraydata(fpointer.read())
                    self.spinden_viewer_2d.arraydata = arrayd
                    self.spinden_viewer_3d.arraydata = arrayd

        self.on_band_change()

        super().__init__([
            self.info,
            ipw.HBox([
                self.bands_viewer,
                ipw.VBox([self.info_out, ipw.HBox([self.orbital_viewer_2d, self.orbital_viewer_3d])],
                         layout=ipw.Layout(margin="200px 0px 0px 0px")),
            ]),
            ipw.HBox([self.spinden_viewer_2d, self.spinden_viewer_3d]),
        ], **kwargs)

    def on_band_change(self, _=None):
        """Replot the orbitals in case of the the band change."""

        # Orbitals_calcs might use fewer nkpoints than bands_calc.
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
                list_of_calcs += [(x.name, orbitals_calc) for x in orbitals_calc.outputs.retrieved.list_objects()]
        for fname, orbitals_calc in sorted(list_of_calcs, key=lambda x: x[0]):
            mtch = re.match(r".*_K(\d\d\d)_B(\d\d\d).*", fname)
            if mtch:
                kpnt, bnd = int(mtch.group(1)), int(mtch.group(2))
                if bnd != self.bands_viewer.selected_band + 1:
                    continue
                if lower < kpnt <= upper:
                    if fname.startswith('output_data_multiple'):
                        self.selected_cube_files.append(orbitals_calc.outputs[fname])
                    else:
                        self.selected_cube_files.append(orbitals_calc.outputs.retrieved.open(fname).name)

        self.info_out.value = "Found {} cube files".format(len(self.selected_cube_files))
        self.on_kpoint_change(None)

    def on_kpoint_change(self, _=None):
        """Replot the orbitals in case of the kpoint change."""
        if self.bands_viewer.selected_kpoint > len(self.selected_cube_files):
            print("Found no cube files")
            self.orbital_viewer_3d.arraydata = None
            self.orbital_viewer_2d.arraydata = None
        else:
            if isinstance(self.selected_cube_files[self.bands_viewer.selected_kpoint - 1], ArrayData):
                arraydata = self.selected_cube_files[self.bands_viewer.selected_kpoint - 1]
            else:
                absfn = self.selected_cube_files[self.bands_viewer.selected_kpoint - 1]
                with gzip.open(absfn) as fpointer:
                    arraydata = from_cube_to_arraydata(fpointer.read())
            self.orbital_viewer_2d.arraydata = arraydata
            self.orbital_viewer_3d.arraydata = arraydata

    @property
    def spindensity_calc(self):
        """Return spindensity plot calculation if present, otherwise return None."""
        try:
            return get_calc_by_label(self._workcalc, "export_spinden")
        except AssertionError:
            return None
