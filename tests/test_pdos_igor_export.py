from types import SimpleNamespace

import numpy as np
import pytest

from nanoribbon.viewers.igor import create_igor_text
from nanoribbon.viewers.pdos_computed import NanoribbonPDOSWidget


class FakeExtras:
    def __init__(self, values):
        self.values = values

    def get(self, key, default=None):
        return self.values.get(key, default)


def make_fake_widget(selected_atoms=None):
    calls = []

    def calc_pdos(**kwargs):
        calls.append(kwargs)
        energy = np.array([-1.0, 0.0])
        if kwargs.get("atmwfcs") is not None:
            return energy, np.array([[0.25], [0.5]])
        return energy, np.array([[1.0], [2.0]])

    widget = SimpleNamespace(
        _workcalc=SimpleNamespace(
            uuid="12345678-1234-5678-1234-567812345678",
            base=SimpleNamespace(extras=FakeExtras({"fermi_energy": -0.25})),
        ),
        sigma_slider=SimpleNamespace(value=0.06),
        ngauss_slider=SimpleNamespace(value=0),
        emin_box=SimpleNamespace(value=-1.0),
        emax_box=SimpleNamespace(value=1.0),
        selected_atoms=set() if selected_atoms is None else set(selected_atoms),
        atmwfc2atom={1: 2},
        kpts=np.array([0.0, 0.5]),
        ase_struct=SimpleNamespace(
            cell=SimpleNamespace(lengths=lambda: np.array([2.0, 10.0, 10.0])),
            get_chemical_formula=lambda: "C12H2",
        ),
        bands=np.array([[[-0.5, 2.0], [0.5, 2.5]]]),
        vacuum_level=0.0,
        nspins=1,
        nbands=2,
        calc_pdos=calc_pdos,
    )
    widget._selected_atmwfcs = lambda: NanoribbonPDOSWidget._selected_atmwfcs(widget)
    return widget, calls


def test_create_igor_text_uses_documented_multiblock_grammar():
    result = create_igor_text(
        [
            (["k", "band"], [np.array([0.0, 0.5]), np.array([-1.0, 1.0])]),
            (["energy", "dos"], [np.array([-1.0, 0.0]), np.array([2.0, 3.0])]),
        ],
        comments=["bands and DOS"],
    )

    assert result.startswith("IGOR\rX // bands and DOS\rWAVES/D\t")
    assert result.endswith("END\r\r")
    assert result.count("\rBEGIN\r") == 2
    assert result.count("\rEND\r") == 2
    assert "\r//" not in result


@pytest.mark.parametrize(
    "wave_blocks",
    [
        [(["bad-name"], [np.array([1.0])])],
        [(["a", "b"], [np.array([1.0]), np.array([1.0, 2.0])])],
    ],
)
def test_create_igor_text_rejects_invalid_blocks(wave_blocks):
    with pytest.raises(ValueError):
        create_igor_text(wave_blocks)


def test_create_figure_builds_combined_bands_and_pdos_plot():
    widget, _calls = make_fake_widget()
    widget.plot_bands = lambda ax, **_kwargs: ax.plot([0.0, 0.5], [-0.5, 0.5])
    widget.plot_pdos = lambda ax, **_kwargs: ax.plot([0.0, 1.0], [-1.0, 1.0])

    figure = NanoribbonPDOSWidget.create_figure(widget)

    assert len(figure.axes) == 2
    assert tuple(figure.get_size_inches()) == pytest.approx((12.0, 8.0))


def test_bands_pdos_export_uses_current_plot_state():
    widget, calls = make_fake_widget()

    result = NanoribbonPDOSWidget.igor_bands_pdos(widget)

    assert calls == [{"ngauss": 0, "sigma": 0.06, "emin": -1.0, "emax": 1.0}]
    assert "X // Viewer energy window: -1 to 1 eV" in result
    assert "X // DOS broadening parameter: 0.06 eV" in result
    assert "nr123456_b0p06_n0_s0r000" in result
    assert "nr123456_b0p06_n0_s0v000" in result
    assert "nr123456_b0p06_n0_s0v001" not in result
    assert "nr123456_b0p06_n0_Erel" in result
    assert "nr123456_b0p06_n0_Evac" in result
    assert "nr123456_b0p06_n0_dos0" in result
    assert "nr123456_b0p06_n0_sel0" not in result


def test_bands_pdos_export_includes_selected_atom_pdos():
    widget, calls = make_fake_widget(selected_atoms={1})

    result = NanoribbonPDOSWidget.igor_bands_pdos(widget)

    assert calls[1]["atmwfcs"] == [0]
    assert "X // Selected atom indices (one-based): 2" in result
    assert "nr123456_b0p06_n0_sel0" in result


@pytest.mark.parametrize("fermi", [-0.25, None])
def test_two_spin_export_preserves_the_numeric_energy_references(fermi):
    widget, _calls = make_fake_widget()
    widget.nspins = 2
    widget.vacuum_level = 4.0
    widget.bands = np.array([[[3.5, 6.0], [4.5, 6.5]], [[3.2, 6.0], [4.8, 6.5]]])
    widget._workcalc.base.extras.values["fermi_energy"] = fermi
    widget.calc_pdos = lambda **kwargs: (
        np.array([-1.0, 0.0]),
        np.array([[1.0, 3.0], [2.0, 4.0]]),
    )
    result = NanoribbonPDOSWidget.igor_bands_pdos(widget)
    blocks = result.split("WAVES/D\t")[1:]
    waves = {}
    for block in blocks:
        header, rows = block.split("\rBEGIN\r")
        values = np.array(
            [
                [float(value) for value in row.split()]
                for row in rows.split("\rEND")[0].splitlines()
            ]
        )
        waves.update(zip(header.split("\t"), values.T, strict=True))
    prefix = "nr123456_b0p06_n0"
    for spin in (0, 1):
        vacuum = widget.bands[spin, :, 0] - widget.vacuum_level
        np.testing.assert_allclose(waves[f"{prefix}_s{spin}v000"], vacuum)
        if fermi is None:
            assert f"{prefix}_s{spin}r000" not in waves
        else:
            np.testing.assert_allclose(waves[f"{prefix}_s{spin}r000"], vacuum - fermi)
    np.testing.assert_allclose(waves[f"{prefix}_dos0"], [1, 2])
    np.testing.assert_allclose(waves[f"{prefix}_dos1"], [3, 4])
