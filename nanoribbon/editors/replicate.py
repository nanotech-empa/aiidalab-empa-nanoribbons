import numpy as np
from numpy.linalg import norm
from ase import Atoms
from ase.data import covalent_radii
from ase.neighborlist import NeighborList
import ase.neighborlist
import scipy.stats
from scipy.constants import physical_constants
import itertools
from IPython.display import display, clear_output, HTML
import nglview
import ipywidgets as ipw

from apps.surfaces.widgets import slabs

from traitlets import HasTraits, Instance, Dict, Unicode, dlink, link, observe
from aiidalab_widgets_base import StructureManagerWidget
from apps.surfaces.widgets.ANALYZE_structure import StructureAnalyzer


class RepGnr(ipw.VBox):
    structure = Instance(Atoms, allow_none=True)
    def __init__(self, title=''):
        self.title = title
        self._molecule = None
        self.nx_slider = ipw.IntSlider(description="nx", min=1, max=6, continuous_update=False)    
        
        self.create_bttn = ipw.Button(description="Replicate")
        self.create_bttn.on_click(self.replicate)
        self.info = ipw.HTML('')
        super().__init__(children=[
            ipw.HBox([self.nx_slider, self.create_bttn]),
            self.info,
        ])
        
    def replicate(self, _=None):
        """Create slab and remember the last molecule used."""
        #sa = StructureAnalyzer()
        #sa.structure = self.molecule
        self.info.value = ''
        atoms = self.structure.copy()
        nx = self.nx_slider.value
        
        self.structure = atoms.repeat((self.nx_slider.value, 1, 1))
        
#    @observe('molecule')
#    def on_struct_change(self, change=None):
#        """Selected molecule from structure."""
#
#        if self.molecule:
#            self.nx_slider.value = 1
