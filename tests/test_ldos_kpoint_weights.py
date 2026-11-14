"""The Gamma-X half-path represents each full-zone state exactly once."""

from types import SimpleNamespace

import numpy as np
import pytest

from nanoribbon.viewers.data import ArraySnapshot
from nanoribbon.viewers.show_computed import NanoribbonShowWidget


@pytest.mark.parametrize("nkpoints", [2, 3, 12, 13])
def test_half_path_equals_explicit_time_reversal_zone(nkpoints):
    energies = np.linspace(-0.5, 0.5, nkpoints)
    density = np.arange(1, nkpoints + 1, dtype=float)
    calls = []

    def cube(_spin, kpoint, _band):
        calls.append(kpoint)
        snapshot = ArraySnapshot()
        snapshot.set_array("data", np.full((2, 2, 2), density[kpoint]))
        return snapshot, f"K{kpoint + 1:03}_B001"

    model = SimpleNamespace(
        bands_lowres=energies[None, :, None],
        vacuum_level=0.0,
        _read_arraydata=cube,
        gaussian=lambda energy, width: NanoribbonShowWidget.gaussian(
            None, energy, width
        ),
        clamp_arraydata_to_zero=lambda snapshot: NanoribbonShowWidget.clamp_arraydata_to_zero(
            None, snapshot
        ),
    )
    result, missing = NanoribbonShowWidget.calculate_sts_mapping(model, 0.0, 0.8)
    # Build one full BZ explicitly: Gamma -> X -> -k interiors -> Gamma,
    # omitting the equivalent periodic endpoint a second time.
    full_indices = np.r_[np.arange(nkpoints), np.arange(nkpoints - 2, 0, -1)]
    sigma = 0.8 / 2.3548
    expected = np.sum(
        density[full_indices]
        * np.exp(-energies[full_indices] ** 2 / (2 * sigma**2))
        / (sigma * np.sqrt(2 * np.pi))
    )
    np.testing.assert_allclose(result.get_array("data"), expected, rtol=2e-15)
    assert not missing
    assert calls == list(range(nkpoints))


@pytest.mark.parametrize("nkpoints", [3, 12, 13])
@pytest.mark.parametrize("kpoint_kind", ["Gamma", "interior", "X"])
def test_endpoint_and_interior_multiplicities(nkpoints, kpoint_kind):
    kpoint = {"Gamma": 0, "interior": 1, "X": nkpoints - 1}[kpoint_kind]
    energies = np.full(nkpoints, 10.0)
    energies[kpoint] = 0.0
    original = np.array([[[2.0, -1.0]]])
    snapshot = ArraySnapshot()
    snapshot.set_array("data", original)
    model = SimpleNamespace(
        bands_lowres=energies[None, :, None],
        vacuum_level=0.0,
        _read_arraydata=lambda *_: (snapshot, "selected"),
        gaussian=lambda energy, width: NanoribbonShowWidget.gaussian(
            None, energy, width
        ),
        clamp_arraydata_to_zero=lambda value: NanoribbonShowWidget.clamp_arraydata_to_zero(
            None, value
        ),
    )
    result, missing = NanoribbonShowWidget.calculate_sts_mapping(model, 0.0, 0.1)
    multiplicity = 2 if kpoint_kind == "interior" else 1
    expected = multiplicity / ((0.1 / 2.3548) * np.sqrt(2 * np.pi))
    np.testing.assert_allclose(result.get_array("data"), [[[2 * expected, 0]]])
    np.testing.assert_array_equal(snapshot.get_array("data"), original)
    assert not missing
