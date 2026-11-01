"""Plain display data that can outlive the AiiDA operation that reads it."""

from dataclasses import dataclass

import numpy as np
from aiida import orm
from ase import Atoms


class ArraySnapshot:
    def __init__(self):
        self._arrays = {}

    @classmethod
    def from_data(cls, value):
        if value is None or isinstance(value, cls):
            return value
        if getattr(value, "is_stored", False):
            value = orm.load_node(value.uuid)
        result = cls()
        for name in value.get_arraynames():
            result.set_array(name, value.get_array(name))
        return result

    def get_arraynames(self):
        return list(self._arrays)

    def get_array(self, name):
        return self._arrays[name]

    def set_array(self, name, value):
        array = np.array(value, copy=True)
        array.setflags(write=False)
        self._arrays[name] = array


@dataclass
class StructureSnapshot:
    atoms: Atoms
    pk: int | None

    @classmethod
    def from_data(cls, value):
        if getattr(value, "is_stored", False):
            value = orm.load_node(value.uuid)
        return cls(value.get_ase(), value.pk)

    @property
    def cell_lengths(self):
        return self.atoms.cell.lengths()

    def get_ase(self):
        return self.atoms.copy()
