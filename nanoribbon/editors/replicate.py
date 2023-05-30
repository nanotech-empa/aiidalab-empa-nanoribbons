import ipywidgets as ipw
from ase import Atoms
from traitlets import Instance


class NanoribbonReplicateEditor(ipw.VBox):
    structure = Instance(Atoms, allow_none=True)

    def __init__(self, title=""):
        self.title = title
        self._molecule = None
        self.nx_slider = ipw.IntSlider(
            description="nx", min=1, max=6, continuous_update=False
        )

        self.create_bttn = ipw.Button(description="Replicate")
        self.create_bttn.on_click(self.replicate)
        self.info = ipw.HTML("")
        super().__init__(
            children=[
                ipw.HBox([self.nx_slider, self.create_bttn]),
                self.info,
            ]
        )

    def replicate(self, _=None):
        """Create slab and remember the last molecule used."""
        # sa = StructureAnalyzer()
        # sa.structure = self.molecule
        self.info.value = ""
        atoms = self.structure.copy()
        self.nx_slider.value

        self.structure = atoms.repeat((self.nx_slider.value, 1, 1))
