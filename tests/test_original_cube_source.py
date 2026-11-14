"""Original-source recovery preserves the existing immutable result."""

import hashlib
import io
from types import SimpleNamespace

import ipywidgets as ipw
import numpy as np
import pytest
from aiida import orm
from aiida.common.links import LinkType

from nanoribbon.viewers.cube_sources import read_original_orbital
from nanoribbon.viewers.data import ArraySnapshot
from nanoribbon.viewers.show_computed import NanoribbonShowWidget

pytest_plugins = ["aiida.tools.pytest_fixtures"]


def cube_text():
    values = " ".join(str(x) for x in range(4 * 8 * 10))
    return (
        "orbital density\noriginal QE cube\n1 0 0 0\n"
        "4 1 0 0\n8 0 1 0\n10 0 0 1\n6 0 1 4 5\n" + values + "\n"
    )


@pytest.fixture
def original(aiida_profile_clean, aiida_localhost, tmp_path):
    calculation = orm.CalcJobNode(
        computer=aiida_localhost, process_type="aiida.calculations:quantumespresso.pp"
    ).store()
    retrieved = orm.FolderData()
    retrieved.base.links.add_incoming(calculation, LinkType.CREATE, "retrieved")
    retrieved.store()
    remote = orm.RemoteData(computer=aiida_localhost, remote_path=str(tmp_path))
    remote.base.links.add_incoming(calculation, LinkType.CREATE, "remote_folder")
    remote.store()
    stored = orm.ArrayData()
    stored.set_array("data", np.full((2, 2, 2), -7.0))
    stored.set_array("data_units", np.array("e/bohr^3"))
    stored.base.links.add_incoming(calculation, LinkType.CREATE, "output_data")
    stored.store()
    path = tmp_path / "aiida.filplot_K001_B015aiida.fileout"
    path.write_text(cube_text())
    return stored.uuid, calculation.uuid, path


def test_original_remote_read_uses_public_local_transport_without_storing(original):
    array_uuid, calc_uuid, path = original
    before_nodes = orm.QueryBuilder().append(orm.Node).count()
    stored_before = orm.load_node(array_uuid).get_array("data")
    raw = path.read_bytes()
    result, provenance = read_original_orbital(array_uuid, "K001_B015")
    assert isinstance(result, ArraySnapshot)
    assert not any(isinstance(x, orm.Node) for x in vars(result).values())
    assert result.get_array("data").min() >= 0
    assert not result.get_array("data").flags.writeable
    assert provenance == {
        "calculation_uuid": calc_uuid,
        "filename": path.name,
        "source": "original work directory",
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    assert orm.QueryBuilder().append(orm.Node).count() == before_nodes
    np.testing.assert_array_equal(
        orm.load_node(array_uuid).get_array("data"), stored_before
    )
    assert path.read_bytes() == raw


def test_retrieved_original_does_not_open_remote_transport(original, monkeypatch):
    array_uuid, calc_uuid, path = original
    # The immutable retrieved folder cannot be edited: attach a fresh test copy
    # to a second PP calculation, using the ordinary CREATE link contract.
    original_calc = orm.load_node(calc_uuid)
    calculation = orm.CalcJobNode(computer=original_calc.computer).store()
    retrieved = orm.FolderData()
    retrieved.base.repository.put_object_from_filelike(
        io.BytesIO(path.read_bytes()), path.name
    )
    retrieved.base.links.add_incoming(calculation, LinkType.CREATE, "retrieved")
    retrieved.store()
    monkeypatch.setattr(
        orm.Computer, "get_transport", lambda *_: pytest.fail("remote access")
    )
    result, provenance = read_original_orbital(calculation.uuid, "K001_B015")
    assert provenance["source"] == "retrieved"
    assert isinstance(result, ArraySnapshot)


def test_missing_original_is_reported_without_falling_back_to_old_array(original):
    array_uuid, _, path = original
    path.unlink()
    with pytest.raises(OSError):
        read_original_orbital(array_uuid, "K001_B015")


@pytest.mark.parametrize("label", ["../K001_B015", "K001_B015/../../x", "K1_B2", ""])
def test_invalid_filename_is_rejected_before_loading_nodes(label, monkeypatch):
    monkeypatch.setattr(orm, "load_node", lambda *_: pytest.fail("node read"))
    with pytest.raises(ValueError, match="single exported orbital"):
        read_original_orbital("unused", label)


def test_orbital_selection_uses_same_original_source_for_orbitals_and_ldos(original):
    array_uuid, _, _ = original
    widget = SimpleNamespace(
        nspin=1,
        nkpoints_lowres=12,
        list_of_calcs=[("K001_B015", array_uuid)],
        orbital_source=SimpleNamespace(value="original"),
    )
    result, name = NanoribbonShowWidget._read_arraydata(widget, 0, 0, 14)
    assert name == "K001_B015"
    assert isinstance(result, ArraySnapshot)
    assert widget._original_cube_provenance["filename"].startswith(
        "aiida.filplot_K001_B015"
    )
    widget.orbital_source.value = "stored"
    stored, _ = NanoribbonShowWidget._read_arraydata(widget, 0, 0, 14)
    np.testing.assert_array_equal(stored.get_array("data"), -7.0)


def test_source_change_clears_old_ldos_export_and_requests_fresh_orbital():
    calls = []
    widget = SimpleNamespace(
        sts_mapping_viewer_wrapper=ipw.VBox([ipw.HTML("old map")]),
        sts_mapping_viewer=SimpleNamespace(dl_link=ipw.HTML("old export")),
        on_kpoint_change=lambda: calls.append("reload"),
    )
    NanoribbonShowWidget._on_orbital_source_change(widget)
    assert not widget.sts_mapping_viewer_wrapper.children
    assert widget.sts_mapping_viewer.dl_link.value == ""
    assert calls == ["reload"]


def test_missing_original_clears_visible_ldos_and_stale_download():
    def missing(*_):
        message = "retained cube removed"
        raise FileNotFoundError(message)

    widget = SimpleNamespace(
        calculate_sts_mapping=missing,
        sts_energy_text=SimpleNamespace(value=-1.0),
        sts_fwhm_text=SimpleNamespace(value=0.1),
        sts_mapping_viewer_wrapper=ipw.VBox([ipw.HTML("old map")]),
        sts_mapping_viewer=SimpleNamespace(dl_link=ipw.HTML("old export")),
    )
    NanoribbonShowWidget.on_sts_btn_click(widget)
    assert (
        "Original cube unavailable"
        in widget.sts_mapping_viewer_wrapper.children[0].value
    )
    assert widget.sts_mapping_viewer.dl_link.value == ""
