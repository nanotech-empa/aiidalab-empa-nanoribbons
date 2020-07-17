import numpy as np
from numpy.linalg import norm

from IPython.display import clear_output
import ipywidgets as ipw
import nglview

from traitlets import Instance

from ase import Atoms
from ase.data import covalent_radii
from ase.neighborlist import NeighborList
import ase.neighborlist

from rdkit.Chem import AllChem
from rdkit.Chem.rdmolfiles import MolFromSmiles


class Smiles2GNRWidget(ipw.VBox):
    """Conver SMILES into 3D structure."""
    structure = Instance(Atoms, allow_none=True)
    SPINNER = """<i class="fa fa-spinner fa-pulse" style="color:red;" ></i>"""

    def __init__(self, title="Smiles to GNR"):
        try:
            import openbabel  # pylint: disable=unused-import
        except ImportError:
            super().__init__(
                [ipw.HTML("The SmilesWidget requires the OpenBabel library, "
                          "but the library was not found.")])
            return
        self.title = title
        self.original_structure = None
        self.selection = set()
        self.cell_ready = False
        self.smiles = ipw.Text()
        self.create_structure_btn = ipw.Button(description="Convert SMILES", button_style='info')

        def print_hello(change):
            print(change)

        self.create_structure_btn.on_click(self._on_button_pressed)
        #self.create_structure_btn.on_click(print_hello)
        self.create_cell_btn = ipw.Button(description="create GNR", button_style='info', disabled=True)
        self.create_cell_btn.on_click(self._on_button2_pressed)
        self.viewer = nglview.NGLWidget()
        self.viewer.stage.set_parameters(mouse_preset='pymol')
        self.viewer.observe(self._on_picked, names='picked')
        self.select_two = ipw.HTML("")
        self.output = ipw.HTML("")
        self.picked_out = ipw.Output()
        self.button2_out = ipw.Output()
        super().__init__([
            self.smiles,
            ipw.Label(value="e.g. C1(C2=CC=C(C3=CC=CC=C3)C=C2)=CC=CC=C1"), self.create_structure_btn, self.select_two,
            self.viewer, self.picked_out, self.output, self.create_cell_btn, self.button2_out
        ])

    @staticmethod
    def guess_scaling_factor(atoms):
        import numpy as np
        from numpy.linalg import norm
        from scipy.stats import mode
        from ase import Atoms
        # set bounding box as cell
        cx = 1.5 * (np.amax(atoms.positions[:, 0]) - np.amin(atoms.positions[:, 0]))
        cy = 1.5 * (np.amax(atoms.positions[:, 1]) - np.amin(atoms.positions[:, 1]))
        cz = 15.0
        atoms.cell = (cx, cy, cz)
        atoms.pbc = (True, True, True)

        # calculate all atom-atom distances
        c_atoms = [a for a in atoms if a.symbol[0] == "C"]
        n = len(c_atoms)
        dists = np.zeros([n, n])
        for i, a in enumerate(c_atoms):
            for j, b in enumerate(c_atoms):
                dists[i, j] = norm(a.position - b.position)

        # find bond distances to closest neighbor
        dists += np.diag([np.inf] * n)  # don't consider diagonal
        bonds = np.amin(dists, axis=1)

        # average bond distance
        avg_bond = float(mode(bonds)[0])

        # scale box to match equilibrium carbon-carbon bond distance
        cc_eq = 1.4313333333
        s = cc_eq / avg_bond
        return s

    @staticmethod
    def scale(atoms, s):
        from ase import Atoms
        cx, cy, cz = atoms.cell
        atoms.set_cell((s * cx, s * cy, cz), scale_atoms=True)
        atoms.center()
        return atoms

    @staticmethod
    def smiles2D(smiles):

        mol = MolFromSmiles(smiles)

        AllChem.Compute2DCoords(mol)
        # get the 2D coordinates

        for c in mol.GetConformers():
            coords = c.GetPositions()

        # get the atom labels
        ll = []
        for i in mol.GetAtoms():
            #ll.append(i.GetSymbol())
            ll.append(i.GetAtomicNum())
        ll = np.asarray(ll)

        # create an ASE frame
        c = Atoms('{:d}N'.format(len(coords)))
        c.set_positions(coords)
        c.set_atomic_numbers(ll)
        return c

    @staticmethod
    def construct_cell(atoms, id1, id2):

        p1 = [atoms[id1].x, atoms[id1].y]
        p0 = [atoms[id2].x, atoms[id2].y]
        p2 = [atoms[id2].x, atoms[id1].y]

        v0 = np.array(p0) - np.array(p1)
        v1 = np.array(p2) - np.array(p1)

        angle = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))

        cx = norm(v0)

        if np.abs(angle) > 0.01:
            atoms.rotate_euler(center=atoms[id1].position, phi=-angle, theta=0.0, psi=0.0)

        yrange = np.amax(atoms.positions[:, 1]) - np.amin(atoms.positions[:, 1])
        zrange = np.amax(atoms.positions[:, 2]) - np.amin(atoms.positions[:, 2])
        cy = 15.0 + yrange
        cz = 15.0 + zrange

        atoms.cell = (cx, cy, cz)
        atoms.pbc = (True, True, True)
        atoms.center()
        atoms.wrap(eps=0.001)

        #### REMOVE REDUNDANT ATOMS
        tobedel = []

        cov_radii = [covalent_radii[a.number] for a in atoms]
        nl = NeighborList(cov_radii, bothways=False, self_interaction=False)
        nl.update(atoms)

        for a in atoms:
            indices, offsets = nl.get_neighbors(a.index)
            for i, offset in zip(indices, offsets):
                dist = norm(a.position - (atoms.positions[i] + np.dot(offset, atoms.get_cell())))
                if dist < 0.4:
                    tobedel.append(atoms[i].index)

        del atoms[tobedel]

        #### ENDFIND UNIT CELL AND APPLIES IT

        #### ADD Hydrogens
        cov_radii = [covalent_radii[a.number] for a in atoms]
        nl = NeighborList(cov_radii, bothways=True, self_interaction=False)
        nl.update(atoms)

        need_a_H = []
        for a in atoms:
            nlist = nl.get_neighbors(a.index)[0]
            if len(nlist) < 3:
                if a.symbol == 'C':
                    need_a_H.append(a.index)

        print("Added missing Hydrogen atoms: ", need_a_H)

        dCH = 1.1
        for a in need_a_H:
            vec = np.zeros(3)
            indices, offsets = nl.get_neighbors(atoms[a].index)
            for i, offset in zip(indices, offsets):
                vec += -atoms[a].position + (atoms.positions[i] + np.dot(offset, atoms.get_cell()))
            vec = -vec / norm(vec) * dCH
            vec += atoms[a].position
            htoadd = ase.Atom('H', vec)
            atoms.append(htoadd)

        return atoms

    def _on_picked(self, _=None):
        """When an attom is picked."""

        self.cell_ready = False

        if 'atom1' not in self.viewer.picked.keys():
            return  # did not click on atom
        self.create_cell_btn.disabled = True

        with self.picked_out:
            clear_output()

            self.viewer.component_0.remove_ball_and_stick()  # pylint: disable=no-member
            self.viewer.component_0.remove_ball_and_stick()  # pylint: disable=no-member
            self.viewer.add_ball_and_stick()  # pylint: disable=no-member

            idx = self.viewer.picked['atom1']['index']

            # Toggle.
            if idx in self.selection:
                self.selection.remove(idx)
            else:
                self.selection.add(idx)

            if len(self.selection) == 2:
                self.create_cell_btn.disabled = False

            #if(selection):
            sel_str = ",".join([str(i) for i in sorted(self.selection)])
            print("Selected atoms: " + sel_str)
            self.viewer.add_representation('ball+stick', selection="@" + sel_str, color='red', aspectRatio=3.0)
            self.viewer.picked = {}  # reset, otherwise immidiately selecting same atom again won't create change event

    def _on_button_pressed(self, _=None):
        """Convert SMILES to ase structure when button is pressed."""
        self.output.value = ""
        self.select_two.value = '<h3>Select two equivalent atoms that define the basis vector</h3>'
        self.create_cell_btn.disabled = True
        if not self.smiles.value:
            return

        smiles = self.smiles.value.replace(" ", "")
        struct = self.smiles2D(smiles)
        self.original_structure = struct.copy()
        if hasattr(self.viewer, "component_0"):
            self.viewer.component_0.remove_ball_and_stick()  # pylint: disable=no-member
            cid = self.viewer.component_0.id  # pylint: disable=no-member
            self.viewer.remove_component(cid)

        # Empty selection.
        self.selection = set()
        self.cell_ready = False

        # Add new component.
        self.viewer.add_component(nglview.ASEStructure(struct))  # adds ball+stick
        self.viewer.center()
        self.viewer.handle_resize()

    def _on_button2_pressed(self, _=None):
        """Generate GNR button pressed."""
        with self.button2_out:
            clear_output()
            self.cell_ready = False

            if len(self.selection) != 2:
                print("You must select exactly two atoms")
                return

            id1 = sorted(self.selection)[0]
            id2 = sorted(self.selection)[1]
            self.structure = self.construct_cell(self.original_structure, id1, id2)
