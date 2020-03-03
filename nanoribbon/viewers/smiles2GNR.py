#import numpy as np
#from scipy.stats import mode
#from numpy.linalg import norm
#from pysmiles import read_smiles,write_smiles
#from rdkit.Chem.rdmolfiles import MolFromSmiles,MolToMolFile
#import networkx as nx
#import math
#from ase import Atoms
#from ase.visualize import view
from IPython.display import display, clear_output
import ipywidgets as ipw
import nglview
#from ase.data import covalent_radii
#from ase.neighborlist import NeighborList
#import ase.neighborlist
import numpy as np
from numpy.linalg import norm

from ase import Atoms
from ase.data import covalent_radii
from ase.neighborlist import NeighborList
import ase.neighborlist


from rdkit.Chem import AllChem
from rdkit.Chem.rdmolfiles import MolFromSmiles,MolToMolFile
        
class Smiles2GNRWidget(ipw.VBox):
    """Conver SMILES into 3D structure."""

    SPINNER = """<i class="fa fa-spinner fa-pulse" style="color:red;" ></i>"""

    def __init__(self):
        try:
            import openbabel  # pylint: disable=unused-import
        except ImportError:
            super().__init__(
                [ipw.HTML("The SmilesWidget requires the OpenBabel library, "
                          "but the library was not found.")])
            return

        self.selection = set()
        self.cell_ready = False
        self.smiles = ipw.Text()
        self.create_structure_btn = ipw.Button(description="Convert SMILES", button_style='info')
        def print_hello(change):
            print(change)
        self.create_structure_btn.on_click(self._on_button_pressed)
        #self.create_structure_btn.on_click(print_hello)
        self.create_cell_btn = ipw.Button(description="create GNR", button_style='info')
        self.create_cell_btn.on_click(self._on_button2_pressed)
        self.viewer = nglview.NGLWidget()
        self.viewer.observe(self._on_picked, names='picked')
        self.output = ipw.HTML("")
        self.picked_out = ipw.Output()
        self.button2_out = ipw.Output()
        super().__init__([self.smiles, self.create_structure_btn, self.viewer, 
                          self.picked_out, self.output,self.create_cell_btn, self.button2_out])
    
########
    @staticmethod
    def guess_scaling_factor(atoms):
        import numpy as np
        from numpy.linalg import norm
        from scipy.stats import mode
        from ase import Atoms
        # set bounding box as cell
        cx = 1.5 * (np.amax(atoms.positions[:,0]) - np.amin(atoms.positions[:,0]))
        cy = 1.5 * (np.amax(atoms.positions[:,1]) - np.amin(atoms.positions[:,1]))
        cz = 15.0
        atoms.cell = (cx, cy, cz)
        atoms.pbc = (True,True,True)

        # calculate all atom-atom distances
        c_atoms = [a for a in atoms if a.symbol[0]=="C"]
        n = len(c_atoms)
        dists = np.zeros([n,n])
        for i, a in enumerate(c_atoms):
            for j, b in enumerate(c_atoms):
                dists[i,j] = norm(a.position - b.position)

        # find bond distances to closest neighbor
        dists += np.diag([np.inf]*n) # don't consider diagonal
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
        atoms.set_cell((s*cx, s*cy, cz), scale_atoms=True)
        atoms.center()
        return atoms
    
    @staticmethod
    def smiles2D(smiles):

        
        mol = MolFromSmiles(smiles)

        AllChem.Compute2DCoords(mol)
        # get the 2D coordinates

        for c in mol.GetConformers():
            coords=c.GetPositions()

        # get the atom labels
        ll=[]
        for i in mol.GetAtoms():
            #ll.append(i.GetSymbol())
            ll.append(i.GetAtomicNum())
        ll=np.asarray(ll)

        # create an ASE frame
        c=Atoms('{:d}N'.format(len(coords)))
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

        angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))

        #angle=np.degrees(angle)

        cx = norm(v0)

        #print np.degrees(angle),v0,v1,p0,p1
        if np.abs(angle) > 0.01:
        #   s.euler_rotate(phi=angle,theta=0,psi=0,center(x[id1],y[id1],z[id1]))
            atoms.rotate_euler(center=atoms[id1].position, phi=-angle,theta=0.0,psi=0.0)

        yrange = np.amax(atoms.positions[:,1])-np.amin(atoms.positions[:,1])
        zrange = np.amax(atoms.positions[:,2])-np.amin(atoms.positions[:,2]) 
        cy = 15.0 + yrange
        cz = 15.0 + zrange    

        atoms.cell = (cx,cy,cz)
        atoms.pbc = (True,True,True)
        atoms.center()
        atoms.wrap(eps=0.001)

        #### REMOVE REDUNDANT ATOMS
        tobedel = []

        cov_radii = [covalent_radii[a.number] for a in atoms]
        nl = NeighborList(cov_radii, bothways = False, self_interaction = False)
        nl.update(atoms)

        for a in atoms:
            indices, offsets = nl.get_neighbors(a.index)
            for i, offset in zip(indices, offsets):
                dist = norm(a.position -(atoms.positions[i] + np.dot(offset, atoms.get_cell())))
                if dist < 0.4 :
                    tobedel.append(atoms[i].index)

        del atoms[tobedel]
        
        #### ENDFIND UNIT CELL AND APPLIES IT

        #### ADD Hydrogens
        cov_radii = [covalent_radii[a.number] for a in atoms]
        nl = NeighborList(cov_radii, bothways = True, self_interaction = False)
        nl.update(atoms)

        need_a_H = []
        for a in atoms:
            nlist=nl.get_neighbors(a.index)[0]
            if len(nlist)<3:
                if a.symbol=='C':
                    need_a_H.append(a.index)

        print("Added missing Hydrogen atoms: ", need_a_H)

        dCH=1.1
        for a in need_a_H:
            vec = np.zeros(3)
            indices, offsets = nl.get_neighbors(atoms[a].index)
            for i, offset in zip(indices, offsets):
                vec += -atoms[a].position +(atoms.positions[i] + np.dot(offset, atoms.get_cell()))
            vec = -vec/norm(vec)*dCH
            vec += atoms[a].position
            htoadd = ase.Atom('H',vec)
            atoms.append(htoadd)

        return atoms    
 
    def _on_picked(self,ca):

        self.cell_ready = False

        if 'atom1' not in self.viewer.picked.keys():
            return # did not click on atom
        with self.picked_out:
            clear_output()

            #viewer.clear_representations()
            self.viewer.component_0.remove_ball_and_stick()
            self.viewer.component_0.remove_ball_and_stick()
            self.viewer.add_ball_and_stick()
            #viewer.add_unitcell()

            idx = self.viewer.picked['atom1']['index']

            # toggle
            if idx in self.selection:
                self.selection.remove(idx)
            else:
                self.selection.add(idx)

            #if(selection):
            sel_str = ",".join([str(i) for i in sorted(self.selection)])
            print("Selected atoms: "+ sel_str)
            self.viewer.add_representation('ball+stick', selection="@"+sel_str, color='red', aspectRatio=3.0)
            #else:
            #    print ("nothing selected")
            self.viewer.picked = {} # reset, otherwise immidiately selecting same atom again won't create change event    
    


    def _on_button_pressed(self, change):  # pylint: disable=unused-argument
        """Convert SMILES to ase structure when button is pressed."""
        self.output.value = ""

        if not self.smiles.value:
            return

        smiles=self.smiles.value.replace(" ", "")
        c=self.smiles2D(smiles)
        # set the cell

        scaling_fac=self.guess_scaling_factor(c)
        scaled_structure=self.scale(c,scaling_fac)
        self.original_structure=c.copy() 
        if hasattr(self.viewer, "component_0"):
            self.viewer.component_0.remove_ball_and_stick()
            #viewer.component_0.remove_unitcell()
            cid = self.viewer.component_0.id
            self.viewer.remove_component(cid)

        # empty selection
        self.selection = set()
        self.cell_ready = False

        # add new component
        self.viewer.add_component(nglview.ASEStructure(c)) # adds ball+stick
        #viewer.add_unitcell()
        self.viewer.center()
                
    def _on_button2_pressed(self, change):        
        with self.button2_out:
            clear_output()
            self.cell_ready = False   

            if len(self.selection) != 2:
                print("You must select exactly two atoms")
                return 


            id1 = sorted(self.selection)[0]
            id2 = sorted(self.selection)[1]
            new_structure = self.construct_cell(self.original_structure, id1, id2)
            formula = new_structure.get_chemical_formula()
            if self.on_structure_selection is not None:
                self.on_structure_selection(structure_ase=new_structure, name=formula)
            self.cell_ready = True

    def on_structure_selection(self, structure_ase, name):
        pass