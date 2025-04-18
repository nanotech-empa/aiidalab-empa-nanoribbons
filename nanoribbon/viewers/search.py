import io
from base64 import b64encode

import ase
import ase.neighborlist
import ipywidgets as ipw
import matplotlib.pyplot as plt
import numpy as np
from aiida import orm
from IPython.display import clear_output

HA2EV = 27.211386245988


class WrongCodeError(Exception):
    """Raised when facing an unexpected code behavior."""

    def __init__(self):
        super().__init__("Something wrong has been implemented. Revise the code!")


class FermiEnergyAndBandsEnergiesError(ValueError):
    """Raised when the Fermi energy is below or above all the bands energies."""

    def __init__(self, where):
        super().__init__(
            "The Fermi energy is {where} the bands energies. I don't know what to do."
        )


class NeedMoreBandsError(ValueError):
    def __init__(self):
        super().__init__(
            "To understand if it is a metal or insulator, I need more bands than number of electrons."
        )


class FermiEnergyOrOccupationsNotPresentError(KeyError):
    def __init__(self):
        super().__init__(
            "I cannot determine metallicity, since I don't have either fermi energy or occupations."
        )


class EitherNumberOfElectronsOrFermiEnergyError(ValueError):
    def __init__(self):
        super().__init__(
            "Specify either the number of electrons or the Fermi energy, but not both."
        )


class FilesNotFoundError(Exception):
    def __init__(self):
        super().__init__(
            "Did not find 'vacuum_hartree.dat' in the repository or 'output_data' array in the output."
        )


class NoAiidaOutError(Exception):
    def __init__(self, label):
        super().__init__(f"Calculation {label} did not retrive aiida.out")


class CalculationNotFinishedCorrectlyError(Exception):
    def __init__(self, label, state):
        super().__init__(f"Calculation {label} in state {state}.")


class WorkchainExceptedError(RuntimeError):
    def __init__(self, workchain):
        super().__init__(
            f"Workchain {workchain.pk} excepted ({workchain.exit_status})."
        )


class NanoribbonSearchWidget(ipw.VBox):
    STYLE = {"description_width": "120px"}
    LAYOUT = ipw.Layout(width="80%")
    PREPROCESS_VERSION = 8.0

    def __init__(self, **kwargs):
        self.inp_pks = ipw.Text(
            description="PKs",
            placeholder="e.g. 4062 4753 (space separated)",
            layout=self.LAYOUT,
            style=self.STYLE,
        )
        self.inp_formula = ipw.Text(
            description="Formulas:",
            placeholder="e.g. C44H16 C36H4",
            layout=self.LAYOUT,
            style=self.STYLE,
        )
        self.text_description = ipw.Text(
            description="Calculation Name: ",
            placeholder="e.g. a name.",
            layout=self.LAYOUT,
            style=self.STYLE,
        )

        def slider(desc, mininum, maximum):
            return ipw.FloatRangeSlider(
                description=desc,
                min=mininum,
                max=maximum,
                value=[mininum, maximum],
                step=0.01,
                layout=self.LAYOUT,
                style=self.STYLE,
            )

        self.inp_gap = slider("Gap:", 0.0, 3.0)
        self.inp_homo = slider("HOMO:", -6.0, -1.0)
        self.inp_lumo = slider("LUMO:", -6.0, -1.0)
        self.inp_efermi = slider("Fermi Energy:", -6.0, -1.0)
        self.inp_tmagn = slider("Total Magn.:", -2.0, 2.0)
        self.inp_amagn = slider("Abs. Magn.:", 0.0, 1.0)
        self.update_filter_limits()

        self.button = ipw.Button(description="Search")
        self.button.on_click(self.on_click)
        self.results = ipw.HTML()
        self.info_out = ipw.Output()

        children = [
            self.inp_pks,
            self.inp_formula,
            self.text_description,
            self.inp_gap,
            self.inp_homo,
            self.inp_lumo,
            self.inp_efermi,
            self.inp_tmagn,
            self.inp_amagn,
            self.button,
            self.results,
            self.info_out,
        ]
        super().__init__(children, **kwargs)

    def on_click(self, change):
        with self.info_out:
            clear_output()
            self.search(do_all=False)  # Info: move to False, when done with the update.

    def preprocess_workchains(self, do_all=False):
        """This function analyzes the new work chains

        :param do_all: (optional) process worchains again independently of
        the value of the preprocess_version parameter (good for the development)
        """
        qb = orm.QueryBuilder()
        filters = {"label": "NanoribbonWorkChain"}
        if not do_all:
            filters["or"] = [
                {"extras": {"!has_key": "preprocess_version"}},
                {"extras.preprocess_version": {"<": self.PREPROCESS_VERSION}},
            ]
        qb.append(orm.WorkChainNode, filters=filters)
        num_preprocessed = 0
        for m in qb.all():  # iterall() would interfere with set_extra()
            n = m[0]
            if not n.is_sealed:
                print(f"Skipping underway workchain PK {n.id}")
                continue
            if "obsolete" not in n.extras:
                n.set_extra("obsolete", False)
            try:
                self.preprocess_one(n)
                n.set_extra("preprocess_successful", True)
                n.set_extra("preprocess_version", self.PREPROCESS_VERSION)
                print(f"Preprocessed PK {n.id}")
                num_preprocessed += 1
            except Exception as e:
                n.set_extra("preprocess_successful", False)
                n.set_extra("preprocess_error", str(e))
                n.set_extra("preprocess_version", self.PREPROCESS_VERSION)
                print(f"Failed to preprocess PK {n.id}: {e}")
        return num_preprocessed

    def preprocess_one(self, workcalc):
        """This function extracts the relevant information about a work chain and puts into extras."""

        def get_calc_by_label(workcalc, label):
            qb = orm.QueryBuilder()
            qb.append(orm.WorkChainNode, filters={"uuid": workcalc.uuid})
            qb.append(
                orm.CalcJobNode,
                with_incoming=orm.WorkChainNode,
                filters={"label": label},
            )
            calc = qb.first()[0]
            return calc

        if workcalc.exit_status or workcalc.is_excepted:
            raise WorkchainExceptedError(workcalc)
        # Formula and description.
        structure = workcalc.inputs.structure
        ase_struct = structure.get_ase()
        workcalc.set_extra("formula", ase_struct.get_chemical_formula())
        workcalc.set_extra("structure_description", structure.description)

        # Thumbnail.
        thumbnail = self.render_thumbnail(ase_struct)
        workcalc.set_extra("thumbnail", thumbnail)

        # Ensure all steps succeed.
        all_steps = [
            "cell_opt1",
            "cell_opt2",
            "scf",
            "export_hartree",
            "bands",
            "export_pdos",
            "bands_lowres",
            "export_orbitals",
        ]

        if "optimize_cell" in workcalc.inputs:
            if not workcalc.inputs.optimize_cell.value:
                all_steps = [
                    "scf",
                    "export_hartree",
                    "bands",
                    "export_pdos",
                    "bands_lowres",
                    "export_orbitals",
                ]

        if any(k.name[-1].isdigit() for k in structure.kinds):
            all_steps.append("export_spinden")

        for label in all_steps:
            calc = get_calc_by_label(workcalc, label)
            if calc.process_state.value != "finished":
                raise CalculationNotFinishedCorrectlyError(
                    label=label, state=calc.process_state.value
                )

            if calc.attributes["output_filename"] not in [
                obj.name for obj in calc.outputs.retrieved.list_objects()
            ]:
                raise NoAiidaOutError(label=label)

            content = calc.outputs.retrieved.get_object_content(
                calc.attributes["output_filename"]
            )

            if "JOB DONE." not in content:
                print(f"Calculation {label} did not print JOB DONE.")

        # Energies.
        scf_calc = get_calc_by_label(workcalc, "scf")
        assert scf_calc.res["fermi_energy_units"] == "eV"
        fermi_energy = scf_calc.res["fermi_energy"]
        assert scf_calc.res["energy_units"] == "eV"
        workcalc.set_extra("total_energy", scf_calc.res["energy"])
        workcalc.set_extra("opt_structure_uuid", scf_calc.inputs.structure.uuid)
        workcalc.set_extra("opt_structure_pk", scf_calc.inputs.structure.pk)

        # Magnetization.
        res = scf_calc.outputs.output_parameters
        abs_mag = res.get_attribute("absolute_magnetization", 0.0)
        workcalc.set_extra(
            "absolute_magnetization_per_angstr", abs_mag / ase_struct.cell[0, 0]
        )
        tot_mag = res.get_attribute("total_magnetization", 0.0)
        workcalc.set_extra(
            "total_magnetization_per_angstr", tot_mag / ase_struct.cell[0, 0]
        )
        energy = res.get_attribute("energy", 0.0)
        workcalc.set_extra("energy", energy)
        workcalc.set_extra("cellx", ase_struct.cell[0, 0])

        # HOMO, LUMO, and Gap.
        bands_calc = get_calc_by_label(workcalc, "bands")
        bands = bands_calc.outputs.output_band
        is_insulator, gap, homo, lumo = self.find_bandgap(
            bands, fermi_energy=fermi_energy
        )
        workcalc.set_extra("is_insulator", is_insulator)
        workcalc.set_extra("gap", gap)

        # Vacuum level.
        export_hartree_calc = get_calc_by_label(workcalc, "export_hartree")
        try:
            fobj = io.StringIO(
                export_hartree_calc.outputs.retrieved.get_object_content(
                    "vacuum_hartree.dat"
                )
            )
            data = np.loadtxt(fobj)[:, 2]
        except FileNotFoundError:
            try:
                data = export_hartree_calc.outputs.output_data.get_array("data")
            except KeyError as exc:
                raise FilesNotFoundError() from exc

        vacuum_level = np.mean(data) * HA2EV * 0.5
        workcalc.set_extra("vacuum_level", vacuum_level)

        # Store shifted energies.
        workcalc.set_extra("fermi_energy", fermi_energy - vacuum_level)
        if is_insulator:
            workcalc.set_extra("homo", homo - vacuum_level)
            workcalc.set_extra("lumo", lumo - vacuum_level)
        else:
            workcalc.set_extra("homo", fermi_energy - vacuum_level)
            workcalc.set_extra("lumo", fermi_energy - vacuum_level)

    @staticmethod
    def render_thumbnail(ase_struct):
        s = ase_struct.repeat((2, 1, 1))
        cov_radii = [ase.data.covalent_radii[a.number] for a in s]
        nl = ase.neighborlist.NeighborList(
            cov_radii, bothways=True, self_interaction=False
        )
        nl.update(s)

        fig, ax = plt.subplots()
        ax.set_aspect(1)
        ax.axes.set_xlim([0, s.cell[0][0]])
        ax.axes.set_ylim([5, s.cell[1][1] - 5])
        ax.set_facecolor((0.85, 0.85, 0.85))
        ax.axes.get_yaxis().set_visible(False)
        ax.tick_params(
            axis="x", which="both", bottom="off", top="off", labelbottom="off"
        )

        for at in s:
            # circles
            x, y, z = at.position
            n = ase.data.atomic_numbers[at.symbol]
            ax.add_artist(
                plt.Circle(
                    (x, y),
                    ase.data.covalent_radii[n] * 0.5,
                    color=ase.data.colors.cpk_colors[n],
                    fill=True,
                    clip_on=True,
                )
            )
            # bonds
            nlist = nl.get_neighbors(at.index)[0]
            for theneig in nlist:
                x, y, z = (s[theneig].position + at.position) / 2
                x0, y0, z0 = at.position
                if (x - x0) ** 2 + (y - y0) ** 2 < 2:
                    ax.plot(
                        [x0, x],
                        [y0, y],
                        color=ase.data.colors.cpk_colors[n],
                        linewidth=2,
                        linestyle="-",
                    )

        img = io.BytesIO()
        fig.savefig(img, format="png", dpi=72, bbox_inches="tight")
        plt.close()
        return b64encode(img.getvalue()).decode()

    @staticmethod
    def find_bandgap(bandsdata, number_electrons=None, fermi_energy=None):
        """
        Tries to guess whether the bandsdata represent an insulator.
        This method is meant to be used only for electronic bands (not phonons)
        By default, it will try to use the occupations to guess the number of
        electrons and find the Fermi Energy, otherwise, it can be provided
        explicitely.
        Also, there is an implicit assumption that the kpoints grid is
        "sufficiently" dense, so that the bandsdata are not missing the
        intersection between valence and conduction band if present.
        Use this function with care!

        :param (float) number_electrons: (optional) number of electrons in the unit cell
        :param (float) fermi_energy: (optional) value of the fermi energy.

        :note: By default, the algorithm uses the occupations array
          to guess the number of electrons and the occupied bands. This is to be
          used with care, because the occupations could be smeared so at a
          non-zero temperature, with the unwanted effect that the conduction bands
          might be occupied in an insulator.
          Prefer to pass the number_of_electrons explicitly

        :note: Only one between number_electrons and fermi_energy can be specified at the
          same time.

        :return: (is_insulator, gap), where is_insulator is a boolean, and gap a
                 float. The gap is None in case of a metal, zero when the homo is
                 equal to the lumo (e.g. in semi-metals).
        """

        def nint(num):
            """
            Stable rounding function
            """
            if num > 0:
                return int(num + 0.5)
            else:
                return int(num - 0.5)

        if fermi_energy and number_electrons:
            raise EitherNumberOfElectronsOrFermiEnergyError()

        assert bandsdata.units == "eV"
        stored_bands = bandsdata.get_bands()

        if len(stored_bands.shape) == 3:
            # I write the algorithm for the generic case of having both the
            # spin up and spin down array

            # put all spins on one band per kpoint
            bands = np.concatenate(list(stored_bands), axis=1)
        else:
            bands = stored_bands

        # analysis on occupations:
        if fermi_energy is None:
            num_kpoints = len(bands)

            if number_electrons is None:
                try:
                    _, stored_occupations = bandsdata.get_bands(also_occupations=True)
                except KeyError as exc:
                    raise FermiEnergyOrOccupationsNotPresentError() from exc

                # put the occupations in the same order of bands, also in case of multiple bands
                if len(stored_occupations.shape) == 3:
                    # I write the algorithm for the generic case of having both the
                    # spin up and spin down array

                    # put all spins on one band per kpoint
                    occupations = np.concatenate(list(stored_occupations), axis=1)
                else:
                    occupations = stored_occupations

                # now sort the bands by energy
                # Note: I am sort of assuming that I have an electronic ground state

                # sort the bands by energy, and reorder the occupations accordingly
                # since after joining the two spins, I might have unsorted stuff
                bands, occupations = (
                    np.array(y)
                    for y in zip(
                        *[
                            zip(*j)
                            for j in [
                                sorted(
                                    zip(i[0].tolist(), i[1].tolist()),
                                    key=lambda x: x[0],
                                )
                                for i in zip(bands, occupations)
                            ]
                        ]
                    )
                )
                number_electrons = int(
                    round(sum([sum(i) for i in occupations]) / num_kpoints)
                )

                homo_indexes = [
                    np.where(np.array([nint(_) for _ in x]) > 0)[0][-1]
                    for x in occupations
                ]
                if (
                    len(set(homo_indexes)) > 1
                ):  # there must be intersections of valence and conduction bands
                    return False, None, None, None
                else:
                    homo = [_[0][_[1]] for _ in zip(bands, homo_indexes)]
                    try:
                        lumo = [_[0][_[1] + 1] for _ in zip(bands, homo_indexes)]
                    except IndexError as exc:
                        raise NeedMoreBandsError() from exc

            else:
                bands = np.sort(bands)
                number_electrons = int(number_electrons)

                # find the zero-temperature occupation per band (1 for spin-polarized
                # calculation, 2 otherwise)
                number_electrons_per_band = 4 - len(stored_bands.shape)  # 1 or 2
                # gather the energies of the homo band, for every kpoint
                homo = [
                    i[number_electrons / number_electrons_per_band - 1] for i in bands
                ]  # take the nth level
                try:
                    # gather the energies of the lumo band, for every kpoint
                    lumo = [
                        i[number_electrons / number_electrons_per_band] for i in bands
                    ]  # take the n+1th level
                except IndexError as exc:
                    raise NeedMoreBandsError() from exc

            if number_electrons % 2 == 1 and len(stored_bands.shape) == 2:
                # if #electrons is odd and we have a non spin polarized calculation
                # it must be a metal and I don't need further checks
                return False, None, None, None

            # if the nth band crosses the (n+1)th, it is an insulator
            gap = min(lumo) - max(homo)
            if gap == 0.0:
                return False, 0.0, None, None
            elif gap < 0.0:
                return False, gap, None, None
            else:
                return True, gap, max(homo), min(lumo)

        # analysis on the fermi energy
        else:
            # reorganize the bands, rather than per kpoint, per energy level

            # I need the bands sorted by energy
            bands.sort()

            levels = bands.transpose()
            max_mins = [(max(i), min(i)) for i in levels]

            if fermi_energy > bands.max():
                raise FermiEnergyAndBandsEnergiesError(where="above")
            if fermi_energy < bands.min():
                raise FermiEnergyAndBandsEnergiesError(where="below")

            # one band is crossed by the fermi energy
            if any(i[1] < fermi_energy and fermi_energy < i[0] for i in max_mins):
                return False, 0.0, None, None

            # case of semimetals, fermi energy at the crossing of two bands
            # this will only work if the dirac point is computed!
            elif any(i[0] == fermi_energy for i in max_mins) and any(
                i[1] == fermi_energy for i in max_mins
            ):
                return False, 0.0, None, None
            # insulating case
            else:
                # Take the max of the band maxima below the fermi energy.
                homo = max([i[0] for i in max_mins if i[0] < fermi_energy])
                # Take the min of the band minima above the fermi energy.x
                lumo = min([i[1] for i in max_mins if i[1] > fermi_energy])

                gap = lumo - homo
                if gap <= 0.0:
                    raise WrongCodeError()
                return True, gap, homo, lumo

    def search(self, do_all=False):
        self.results.value = "preprocessing..."
        n_preprocessed = self.preprocess_workchains(do_all=do_all)

        if n_preprocessed > 0:
            self.update_filter_limits()

        self.results.value = "searching..."

        # html table header
        html = "<style>#aiida_results td,th {padding: 2px}</style>"
        html += '<form action="compare.ipynb" method="get" target="_blank">'
        html += '<table border=1 id="aiida_results" style="margin:10px;"><tr>'
        html += "<th></th>"
        html += "<th>PK</th>"
        html += "<th>Created</th>"
        html += "<th>Formula</th>"
        html += "<th>CalcName</th>"
        html += "<th>HOMO (eV)</th>"
        html += "<th>LUMO (eV)</th>"
        html += "<th>GAP (eV)</th>"
        html += "<th>Fermi (eV)</th>"
        html += "<th>Energy (eV)</th>"
        html += "<th>Cell x (&#8491;)</th>"
        html += "<th>Total Mag./&#x212B;</th>"
        html += "<th>Abs Mag./&#x212B;</th>"
        html += "<th>Charge</th>"
        html += "<th>Structure</th>"
        html += "<th></th>"
        html += "</tr>"

        # Query AiiDA database.
        filters = {}
        filters["label"] = "NanoribbonWorkChain"
        filters["extras.preprocess_version"] = self.PREPROCESS_VERSION
        filters["extras.preprocess_successful"] = True
        filters["extras.obsolete"] = False

        pk_list = self.inp_pks.value.strip().split()
        if pk_list:
            filters["id"] = {"in": pk_list}

        formula_list = self.inp_formula.value.strip().split()
        if self.inp_formula.value:
            filters["extras.formula"] = {"in": formula_list}

        if len(self.text_description.value) > 1:
            filters["description"] = {"like": f"%{self.text_description.value}%"}

        def add_range_filter(bounds, label):
            filters["extras." + label] = {"and": [{">=": bounds[0]}, {"<": bounds[1]}]}

        add_range_filter(self.inp_gap.value, "gap")
        add_range_filter(self.inp_homo.value, "homo")
        add_range_filter(self.inp_lumo.value, "lumo")
        add_range_filter(self.inp_efermi.value, "fermi_energy")
        add_range_filter(self.inp_tmagn.value, "total_magnetization_per_angstr")
        add_range_filter(self.inp_amagn.value, "absolute_magnetization_per_angstr")

        qb = orm.QueryBuilder()
        qb.append(orm.WorkChainNode, filters=filters)
        qb.order_by({orm.WorkChainNode: {"ctime": "desc"}})

        for node_tuple in qb.all():
            node = node_tuple[0]
            thumbnail = node.get_extra("thumbnail")
            node.get_extra("structure_description")
            opt_structure_uuid = node.get_extra("opt_structure_uuid")
            opt_structure_pk = node.get_extra("opt_structure_pk")

            # Append table row.
            html += "<tr>"
            html += f'<td><input type="checkbox" name="pk" value="{node.id}"></td>'
            html += f'<td><a target="_blank" href="./view_geometry.ipynb?uuid={opt_structure_uuid}">{node.pk}</a></td>'
            html += "<td>%s</td>" % node.ctime.strftime("%Y-%m-%d %H:%M")
            html += "<td>%s</td>" % node.get_extra("formula")
            html += "<td>%s</td>" % node.description
            html += "<td>%4.2f</td>" % node.get_extra("homo")
            html += "<td>%4.2f</td>" % node.get_extra("lumo")
            html += "<td>%4.2f</td>" % node.get_extra("gap")
            html += "<td>%4.2f</td>" % node.get_extra("fermi_energy")
            html += "<td>%9.3f</td>" % node.get_extra("energy")
            html += "<td>%5.2f</td>" % node.get_extra("cellx")
            html += "<td>%4.2f</td>" % node.get_extra("total_magnetization_per_angstr")
            html += "<td>%4.2f</td>" % node.get_extra(
                "absolute_magnetization_per_angstr"
            )
            html += "<td>%3.1f</td>" % (
                node.inputs.tot_charge.base.attributes.get("value")
                if "tot_charge" in node.inputs
                else 0.0
            )
            html += f'<td><a target="_blank" href="./view_geometry.ipynb?uuid={opt_structure_uuid}">'
            html += f'<img src="data:image/png;base64,{thumbnail}" title="{opt_structure_pk}"></a></td>'
            html += (
                f'<td><a target="_blank" href="./show.ipynb?id={node.id}">Show</a><br>'
            )
            html += f'<a target="_blank" href="./pdos.ipynb?id={node.id}">PDOS</a></td>'
            html += "</tr>"

        html += "</table>"
        html += f"Found {qb.count()} matching entries.<br>"
        html += '<input type="submit" value="Compare">'
        html += "</form>"

        self.results.value = html

    def update_filter_limits(self):
        """Query the database to initalize the extremal values of the filters."""
        filters = {}
        filters["label"] = "NanoribbonWorkChain"
        filters["extras.preprocess_version"] = self.PREPROCESS_VERSION
        filters["extras.preprocess_successful"] = True
        filters["extras.obsolete"] = False

        qb = orm.QueryBuilder()
        qb.append(orm.WorkChainNode, filters=filters)

        def compare_and_set(val, slider):
            if val < slider.min:
                slider.min = val
            if val > slider.max:
                slider.max = val
            # also reset to min-max
            slider.value = (slider.min, slider.max)

        for m in qb.all():
            calc = m[0]
            compare_and_set(calc.get_extra("gap"), self.inp_gap)
            compare_and_set(calc.get_extra("homo"), self.inp_homo)
            compare_and_set(calc.get_extra("lumo"), self.inp_lumo)
            compare_and_set(calc.get_extra("fermi_energy"), self.inp_efermi)
            compare_and_set(
                calc.get_extra("total_magnetization_per_angstr"), self.inp_tmagn
            )
            compare_and_set(
                calc.get_extra("absolute_magnetization_per_angstr"), self.inp_amagn
            )
