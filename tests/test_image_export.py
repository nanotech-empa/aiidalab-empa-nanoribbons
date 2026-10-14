"""Image downloads preserve the displayed bands, DOS, spins and projections."""

import re
from base64 import b64decode
from types import MethodType, SimpleNamespace
from xml.etree import ElementTree

import ipywidgets as ipw
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from nanoribbon.viewers import pdos_computed
from nanoribbon.viewers.pdos_computed import NanoribbonPDOSWidget
from nanoribbon.viewers.show_computed import BandsViewerWidget


def make_plot_widget(nspins, selected):
    bands = np.tile([[[-1.0, 0.5], [-0.5, 1.0], [-1.0, 0.5]]], (nspins, 1, 1))
    widget = SimpleNamespace(
        sigma_slider=SimpleNamespace(value=0.1),
        ngauss_slider=SimpleNamespace(value=0),
        emin_box=SimpleNamespace(value=-2.0),
        emax_box=SimpleNamespace(value=2.0),
        band_proj_box=SimpleNamespace(value=0.1),
        colorpicker=SimpleNamespace(value="orange"),
        selected_atoms={0} if selected else set(),
        atmwfc2atom={1: 1, 2: 2},
        bands=bands,
        eigvalues=np.swapaxes(bands, 1, 2),
        projections=np.ones((nspins, 2, 3, 2)) / 2,
        kpoint_weights=np.full(3, 1 / 3),
        vacuum_level=0.0,
        homo=-0.5,
        lumo=0.5,
        nspins=nspins,
        nbands=2,
        nkpoints=3,
        ase_struct=SimpleNamespace(get_chemical_formula=lambda: "C2"),
        structure=SimpleNamespace(pk=42),
    )
    for name in ("calc_pdos", "plot_bands", "plot_pdos", "_mk_figure_link"):
        setattr(widget, name, MethodType(getattr(NanoribbonPDOSWidget, name), widget))
    return widget


@pytest.mark.parametrize("nspins", (1, 2))
@pytest.mark.parametrize("selected", (False, True))
def test_combined_svg_is_vector_and_preserves_plot(monkeypatch, nspins, selected):
    widget = make_plot_widget(nspins, selected)
    figure = NanoribbonPDOSWidget.create_figure(widget)
    displayed = []
    monkeypatch.setattr(pdos_computed, "display", displayed.append)
    original_fonttype = matplotlib.rcParams["svg.fonttype"]
    try:
        NanoribbonPDOSWidget.mk_svg_link(widget, figure)
        link = displayed[0].data
        payload = re.search(r"data:image/svg\+xml;base64,([^\"]+)", link).group(1)
        root = ElementTree.fromstring(b64decode(payload))
        ns = {"svg": "http://www.w3.org/2000/svg"}
        labels = ["".join(node.itertext()) for node in root.findall(".//svg:text", ns)]
        assert root.tag == "{http://www.w3.org/2000/svg}svg"
        assert not root.findall(".//svg:image", ns)
        assert root.findall(".//svg:path", ns)
        assert 'download="C2_pk42.svg"' in link
        assert len(figure.axes) == 2 * nspins
        for spin in range(nspins):
            assert f"Spin {spin}" in labels
            assert figure.axes[2 * spin].get_ylim() == (-2.0, 2.0)
            if selected:
                assert figure.axes[2 * spin].collections
        assert matplotlib.rcParams["svg.fonttype"] == original_fonttype
    finally:
        plt.close(figure)


@pytest.mark.parametrize(
    "image_format,mime,signature",
    [("png", "image/png", b"\x89PNG"), ("pdf", "application/pdf", b"%PDF")],
)
def test_existing_image_downloads(monkeypatch, image_format, mime, signature):
    widget = make_plot_widget(1, False)
    figure, axis = plt.subplots()
    axis.plot([0, 1], [1, 0])
    displayed = []
    monkeypatch.setattr(pdos_computed, "display", displayed.append)
    try:
        getattr(NanoribbonPDOSWidget, f"mk_{image_format}_link")(widget, figure)
        link = displayed[0].data
        payload = re.search(f'data:{mime};base64,([^"]+)', link).group(1)
        assert b64decode(payload).startswith(signature)
        assert f'download="C2_pk42.{image_format}"' in link
    finally:
        plt.close(figure)


@pytest.mark.parametrize("spin", (0, 1))
def test_bands_svg_button_exports_current_interactive_figure(monkeypatch, spin):
    widget = SimpleNamespace(
        bands_array=np.zeros((2, 3, 2)),
        homo=-0.5,
        lumo=0.5,
        vacuum_level=0.0,
        structure=SimpleNamespace(cell_lengths=[2.0, 10.0, 10.0]),
        mk_igor_link=lambda _spin: ipw.HTML(),
    )
    box, _bands, _parabola = BandsViewerWidget.plot_bands(widget, spin)
    figure = box.children[0]
    messages = []
    monkeypatch.setattr(figure, "send", messages.append)
    button = next(
        child
        for child in box.children
        if getattr(child, "description", "") == "Download SVG"
    )
    button.click()
    assert messages == [{"type": "save_svg", "filename": f"bands_spin{spin}.svg"}]


def test_plot_all_offers_svg_alongside_existing_downloads(monkeypatch):
    calls = []
    figure = object()
    widget = SimpleNamespace(create_figure=lambda: figure)
    for name in (
        "mk_png_link",
        "mk_pdf_link",
        "mk_svg_link",
        "mk_bands_txt_link",
        "mk_igor_link",
    ):
        setattr(widget, name, lambda *args, name=name: calls.append((name, args)))
    monkeypatch.setattr(pdos_computed.plt, "show", lambda: None)
    NanoribbonPDOSWidget.plot_all(widget)
    assert calls == [
        ("mk_png_link", (figure,)),
        ("mk_pdf_link", (figure,)),
        ("mk_svg_link", (figure,)),
        ("mk_bands_txt_link", ()),
        ("mk_igor_link", ()),
    ]
