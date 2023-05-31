"""Widget to convert CDXML to nanoribbons."""

import re

import ase
import ase.neighborlist
import ipywidgets as ipw
import nglview
import numpy as np
import scipy
import sklearn.decomposition
import traitlets as tl
from IPython.display import clear_output

try:
    import pybel as pb
except ModuleNotFoundError:
    from openbabel import pybel as pb


class CdxmlUpload2GnrWidget(ipw.VBox):
    """Class that allows to upload structures from user's computer."""

    structure = tl.Instance(ase.Atoms, allow_none=True)

    def __init__(self, title="CDXML to GNR", description="Upload Structure"):
        try:
            import openbabel  # noqa: F401
        except ImportError:
            super().__init__(
                [
                    ipw.HTML(
                        "The SmilesWidget requires the OpenBabel library, "
                        "but the library was not found."
                    )
                ]
            )
            return

        self.title = title
        self.mols = None
        self.original_structure = None
        self.selection = set()
        self.file_upload = ipw.FileUpload(
            description=description, multiple=False, layout={"width": "initial"}
        )
        supported_formats = ipw.HTML(
            """<a href="https://pubs.acs.org/doi/10.1021/ja0697875" target="_blank">
        Supported structure formats: ".cdxml"
        </a>"""
        )

        self.file_upload.observe(self._on_file_upload, names="value")

        self.create_cell_btn = ipw.Button(
            description="create GNR", button_style="info", disabled=True
        )
        self.create_cell_btn.on_click(self._on_cell_button_pressed)

        self.viewer = nglview.NGLWidget()
        self.viewer.stage.set_parameters(mouse_preset="pymol")
        self.viewer.observe(self._on_picked, names="picked")
        self.allmols = ipw.Dropdown(
            options=[None], description="Select mol", value=None, disabled=True
        )
        self.allmols.observe(self._on_sketch_selected, names="value")
        self.select_two = ipw.HTML("")
        self.picked_out = ipw.Output()
        self.cell_button_out = ipw.Output()
        super().__init__(
            children=[
                self.file_upload,
                supported_formats,
                self.allmols,
                self.select_two,
                self.viewer,
                self.picked_out,
                self.create_cell_btn,
                self.cell_button_out,
            ]
        )

    @staticmethod
    def guess_scaling_factor(atoms):
        """Scaling factor to correct the bond length."""

        # Set bounding box as cell.
        c_x = 1.5 * (np.amax(atoms.positions[:, 0]) - np.amin(atoms.positions[:, 0]))
        c_y = 1.5 * (np.amax(atoms.positions[:, 1]) - np.amin(atoms.positions[:, 1]))
        c_z = 15.0
        atoms.cell = (c_x, c_y, c_z)
        atoms.pbc = (True, True, True)

        # Calculate all atom-atom distances.
        c_atoms = [a for a in atoms if a.symbol[0] == "C"]
        n_atoms = len(c_atoms)
        dists = np.zeros([n_atoms, n_atoms])
        for i, atom_a in enumerate(c_atoms):
            for j, atom_b in enumerate(c_atoms):
                dists[i, j] = np.linalg.norm(atom_a.position - atom_b.position)

        # Find bond distances to closest neighbor.
        dists += np.diag([np.inf] * n_atoms)  # Don't consider diagonal.
        bonds = np.amin(dists, axis=1)

        # Average bond distance.
        avg_bond = float(scipy.stats.mode(bonds)[0])

        # Scale box to match equilibrium carbon-carbon bond distance.
        cc_eq = 1.4313333333
        return cc_eq / avg_bond

    @staticmethod
    def scale(atoms, s):
        """Scale atomic positions by the `factor`."""
        c_x, c_y, c_z = atoms.cell
        atoms.set_cell((s * c_x, s * c_y, c_z), scale_atoms=True)
        atoms.center()
        return atoms

    @staticmethod
    def pybel2ase(mol):
        """converts pybel molecule into ase.Atoms"""
        species = [ase.data.chemical_symbols[atm.atomicnum] for atm in mol.atoms]
        pos = np.asarray([atm.coords for atm in mol.atoms])
        pca = sklearn.decomposition.PCA(n_components=3)
        posnew = pca.fit_transform(pos)
        atoms = ase.Atoms(species, positions=posnew)
        sys_size = np.ptp(atoms.positions, axis=0)
        atoms.rotate(-90, "z")  # cdxml are rotated
        atoms.pbc = True
        atoms.cell = sys_size + 10
        atoms.center()

        return atoms

    @staticmethod
    def construct_cell(atoms, id1, id2):
        """Construct periodic cell based on two selected equivalent atoms."""

        pos = [
            [atoms[id2].x, atoms[id2].y],
            [atoms[id1].x, atoms[id1].y],
            [atoms[id2].x + int(1), atoms[id1].y],
        ]

        vec = [np.array(pos[0]) - np.array(pos[1]), np.array(pos[2]) - np.array(pos[1])]
        c_x = np.linalg.norm(vec[0])

        angle = np.math.atan2(np.linalg.det([vec[0], vec[1]]), np.dot(vec[0], vec[1]))
        if np.abs(angle) > 0.01:
            atoms.rotate_euler(
                center=atoms[id1].position, phi=-angle, theta=0.0, psi=0.0
            )

        c_y = 15.0 + np.amax(atoms.positions[:, 1]) - np.amin(atoms.positions[:, 1])
        c_z = 15.0 + np.amax(atoms.positions[:, 2]) - np.amin(atoms.positions[:, 2])

        atoms.cell = (c_x, c_y, c_z)
        atoms.pbc = (True, True, True)
        atoms.center()
        atoms.wrap(eps=0.001)

        # Remove redundant atoms. ORIGINAL
        tobedel = []

        n_l = ase.neighborlist.NeighborList(
            [ase.data.covalent_radii[a.number] for a in atoms],
            bothways=False,
            self_interaction=False,
        )
        n_l.update(atoms)

        for atm in atoms:
            indices, offsets = n_l.get_neighbors(atm.index)
            for i, offset in zip(indices, offsets):
                dist = np.linalg.norm(
                    atm.position
                    - (atoms.positions[i] + np.dot(offset, atoms.get_cell()))
                )
                if dist < 0.4:
                    tobedel.append(atoms[i].index)

        del atoms[tobedel]

        # Find unit cell and apply it.

        # Add Hydrogens.
        n_l = ase.neighborlist.NeighborList(
            [ase.data.covalent_radii[a.number] for a in atoms],
            bothways=True,
            self_interaction=False,
        )
        n_l.update(atoms)

        need_hydrogen = []
        for atm in atoms:
            if len(n_l.get_neighbors(atm.index)[0]) < 3:
                if atm.symbol == "C" or atm.symbol == "N":
                    need_hydrogen.append(atm.index)

        print("Added missing Hydrogen atoms: ", need_hydrogen)

        for atm in need_hydrogen:
            vec = np.zeros(3)
            indices, offsets = n_l.get_neighbors(atoms[atm].index)
            for i, offset in zip(indices, offsets):
                vec += -atoms[atm].position + (
                    atoms.positions[i] + np.dot(offset, atoms.get_cell())
                )
            vec = -vec / np.linalg.norm(vec) * 1.1 + atoms[atm].position
            atoms.append(ase.Atom("H", vec))

        return atoms

    def _on_picked(self, _=None):
        """When an attom is picked."""

        if "atom1" not in self.viewer.picked.keys():
            return  # did not click on atom
        self.create_cell_btn.disabled = True

        with self.picked_out:
            clear_output()

            self.viewer.component_0.remove_ball_and_stick()
            self.viewer.component_0.remove_ball_and_stick()
            self.viewer.add_ball_and_stick()

            idx = self.viewer.picked["atom1"]["index"]

            # Toggle.
            if idx in self.selection:
                self.selection.remove(idx)
            else:
                self.selection.add(idx)

            if len(self.selection) == 2:
                self.create_cell_btn.disabled = False

            # if(selection):
            sel_str = ",".join([str(i) for i in sorted(self.selection)])
            print("Selected atoms: " + sel_str)
            self.viewer.add_representation(
                "ball+stick", selection="@" + sel_str, color="red", aspectRatio=3.0
            )
            self.viewer.picked = (
                {}
            )  # reset, otherwise immidiately selecting same atom again won't create change event

    def _on_file_upload(self, change=None):
        """When file upload button is pressed."""
        self.mols = None
        self.create_cell_btn.disabled = True
        listmols = []
        molid = 0

        fname = list(change["new"].keys())[0]
        frmt = fname.split(".")[-1]
        if frmt == "cdxml":
            cdxml_file_string = self.file_upload.value[fname]["content"].decode("ascii")
            self.mols = re.findall(
                "<fragment(.*?)/fragment", cdxml_file_string, re.DOTALL
            )
            for m in self.mols:
                m = pb.readstring("cdxml", "<fragment" + m + "/fragment>")
                self.mols[molid] = m
                listmols.append(
                    (str(molid) + ": " + m.formula, molid)
                )  # m must be a pb object!!!
                molid += 1
            self.allmols.options = listmols
            self.allmols.value = 0
            self.allmols.disabled = False

    def _on_sketch_selected(self, change=None):
        self.structure = None  # needed to empty view in second viewer
        if self.mols is None or self.allmols.value is None:
            return
        self.create_cell_btn.disabled = True
        atoms = self.pybel2ase(self.mols[self.allmols.value])
        factor = self.guess_scaling_factor(atoms)
        struct = self.scale(atoms, factor)
        self.select_two.value = (
            "<h3>Select two equivalent atoms that define the basis vector</h3>"
        )
        self.original_structure = struct.copy()
        if hasattr(self.viewer, "component_0"):
            self.viewer.component_0.remove_ball_and_stick()
            cid = self.viewer.component_0.id
            self.viewer.remove_component(cid)

        # Empty selection.
        self.selection = set()

        # Add new component.
        self.viewer.add_component(nglview.ASEStructure(struct))  # adds ball+stick
        self.viewer.center()
        self.viewer.handle_resize()
        self.file_upload.value.clear()

    def _on_cell_button_pressed(self, _=None):
        """Generate GNR button pressed."""
        self.create_cell_btn.disabled = True
        with self.cell_button_out:
            clear_output()
            if len(self.selection) != 2:
                print("You must select exactly two atoms")
                return

            id1 = sorted(self.selection)[0]
            id2 = sorted(self.selection)[1]
            incoming_struct = self.original_structure.copy()
            self.structure = self.construct_cell(self.original_structure, id1, id2)
            self.original_structure = incoming_struct.copy()

            if hasattr(self.viewer, "component_0"):
                self.viewer.component_0.remove_ball_and_stick()
                cid = self.viewer.component_0.id
                self.viewer.remove_component(cid)
            # Empty selection.
            self.selection = set()

            # Add new component.
            self.viewer.add_component(
                nglview.ASEStructure(self.original_structure)
            )  # adds ball+stick
            self.viewer.center()
            self.viewer.handle_resize()
