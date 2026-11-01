import gzip
import io
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
from aiida import orm
from aiida.common.links import LinkType
from ase import Atoms

from nanoribbon.viewers.data import ArraySnapshot, StructureSnapshot
from nanoribbon.viewers.show_computed import (
    BandsViewerWidget,
    CubeArrayData2dViewerWidget,
    CubeArrayData3dViewerWidget,
    NanoribbonShowWidget,
)

pytest_plugins = ["aiida.tools.pytest_fixtures"]


@pytest.fixture
def cube(aiida_profile_clean):
    node = orm.ArrayData()
    node.set_array("data", np.arange(8.0).reshape(2, 2, 2) - 3)
    node.set_array("voxel", np.eye(3))
    node.set_array("atomic_numbers", np.array([6]))
    node.set_array("coordinates", np.zeros((1, 3)))
    node.set_array("data_units", np.array("e/bohr^3"))
    return node.store()


def close_widgets(widget):
    seen = set()

    def close(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        for child in getattr(value, "children", ()):
            close(child)
        value.close()

    close(widget)


def test_cube_widget_retains_plain_arrays_and_leaves_stored_data_unchanged(
    cube, monkeypatch
):
    monkeypatch.setattr(plt, "show", lambda: None)
    original = cube.get_array("data")
    widget = CubeArrayData2dViewerWidget()
    try:
        widget.arraydata = cube
        assert isinstance(widget.arraydata, ArraySnapshot)
        assert not isinstance(widget.arraydata, orm.Node)
        np.testing.assert_array_equal(widget.selected_data, original)
        widget.height_slider.value = 1
        widget.update_plot()
        clamped = NanoribbonShowWidget.clamp_arraydata_to_zero(None, widget.arraydata)
        np.testing.assert_array_equal(
            clamped.get_array("data"), np.maximum(original, 0)
        )
        np.testing.assert_array_equal(widget.arraydata.get_array("data"), original)
        np.testing.assert_array_equal(cube.get_array("data"), original)
        assert not widget.arraydata.get_array("data").flags.writeable
        widget.arraydata = None
    finally:
        close_widgets(widget)
        plt.close("all")


def test_read_modern_cube_by_uuid_returns_plain_snapshot(cube):
    widget = SimpleNamespace(
        nspin=1, nkpoints_lowres=2, list_of_calcs=[("K001_B001", cube.uuid)]
    )
    result, name = NanoribbonShowWidget._read_arraydata(widget, 0, 0, 0)
    assert name == "K001_B001"
    assert isinstance(result, ArraySnapshot)
    np.testing.assert_array_equal(result.get_array("data"), cube.get_array("data"))


def test_three_dimensional_viewer_accepts_the_same_plain_snapshot(cube):
    widget = CubeArrayData3dViewerWidget()
    try:
        widget.arraydata = cube
        assert isinstance(widget.arraydata, ArraySnapshot)
        np.testing.assert_array_equal(widget.data_3d, cube.get_array("data"))
        assert widget.viewer._ngl_component_ids
    finally:
        close_widgets(widget)


def test_read_legacy_cube_from_repository_bytes(aiida_profile_clean, aiida_localhost):
    calculation = orm.CalcJobNode(computer=aiida_localhost).store()
    folder = orm.FolderData()
    name = "old_K001_B001.cube.gz"
    text = "comment\ncomment\n1 0 0 0\n2 1 0 0\n2 0 1 0\n2 0 0 1\n6 0 0 0 0\n0 1 2 3\n4 5 6 7\n"
    folder.base.repository.put_object_from_filelike(
        io.BytesIO(gzip.compress(text.encode())), name
    )
    folder.base.links.add_incoming(calculation, LinkType.CREATE, "retrieved")
    folder.store()
    widget = SimpleNamespace(
        nspin=1, nkpoints_lowres=2, list_of_calcs=[(name, calculation.uuid)]
    )
    result, _ = NanoribbonShowWidget._read_arraydata(widget, 0, 0, 0)
    np.testing.assert_array_equal(
        result.get_array("data"), np.arange(8).reshape(2, 2, 2)
    )


def test_bands_widget_uses_plain_structure_and_band_data(aiida_profile_clean):
    structure = orm.StructureData(
        ase=Atoms("C", positions=[[0, 0, 0]], cell=[2, 8, 8], pbc=True)
    ).store()
    bands = orm.BandsData()
    bands.set_kpoints(np.zeros((12, 3)))
    values = (
        np.tile(np.array([-3, -2, -1, 1, 2, 3.0]), (12, 1))
        + np.arange(12)[:, None] ** 2 * 0.01
    )
    bands.set_bands(values, units="eV")
    bands.store()
    widget = BandsViewerWidget(
        bands=bands,
        structure=structure,
        vacuum_level=0,
        homo=-1,
        lumo=1,
        nelectrons=6,
        num_export_bands=2,
    )
    try:
        assert isinstance(widget.structure, StructureSnapshot)
        assert not any(isinstance(value, orm.Node) for value in vars(widget).values())
        np.testing.assert_array_equal(widget.bands_array[0], values)
        np.testing.assert_array_equal(widget.structure.cell_lengths, [2, 8, 8])
        np.testing.assert_array_equal(bands.get_bands(), values)
    finally:
        close_widgets(widget)


def test_pdos_viewer_retains_plain_data_and_reloads_export_metadata(
    aiida_profile_clean, monkeypatch
):
    from nanoribbon.viewers import pdos_computed

    structure = orm.StructureData(
        ase=Atoms("C", positions=[[0, 0, 0]], cell=[2, 8, 8], pbc=True)
    ).store()
    bands = orm.BandsData()
    bands.set_kpoints(np.zeros((3, 3)))
    bands.set_bands(np.zeros((3, 2)), units="eV")
    bands.store()
    retrieved = orm.FolderData()
    xml = (
        '<ROOT><HEADER NUMBER_OF_BANDS="2" NUMBER_OF_K-POINTS="3" '
        'NUMBER_OF_SPIN_COMPONENTS="1" NUMBER_OF_ATOMIC_WFC="1"/>'
        "<EIGENSTATES>" + '<K-POINT Weight="0.3333333333"/><E>0\n0.1</E>'
        "<PROJS><ATOMIC_WFC>1,0\n1,0</ATOMIC_WFC></PROJS>" * 3 + "</EIGENSTATES></ROOT>"
    )
    retrieved.base.repository.put_object_from_filelike(
        io.BytesIO(xml.encode()), "atomic_proj.xml"
    )
    retrieved.base.repository.put_object_from_filelike(
        io.BytesIO(b"\n state #   1: atom   1 (C  ), wfc  1 (l=0 m=1)\n"), "aiida.out"
    )
    retrieved.store()
    calculations = {
        "export_pdos": SimpleNamespace(outputs=SimpleNamespace(retrieved=retrieved)),
        "bands": SimpleNamespace(
            inputs=SimpleNamespace(structure=structure),
            outputs=SimpleNamespace(output_band=bands),
        ),
    }
    monkeypatch.setattr(
        pdos_computed, "get_calc_by_label", lambda _node, label: calculations[label]
    )
    workcalc = orm.WorkChainNode().store()
    workcalc.base.extras.set_many({"vacuum_level": 0, "homo": -1, "lumo": 1})
    widget = pdos_computed.NanoribbonPDOSWidget(workcalc)
    try:
        assert isinstance(widget.structure, StructureSnapshot)
        assert not any(isinstance(value, orm.Node) for value in vars(widget).values())
        assert "metadata unavailable" in widget.igor_bands_pdos()
        orm.load_node(workcalc.uuid).base.extras.set("fermi_energy", -0.2)
        assert "Fermi energy relative to vacuum: -0.2 eV" in widget.igor_bands_pdos()
    finally:
        close_widgets(widget)
