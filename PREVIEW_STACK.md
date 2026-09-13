# Nanoribbons v2.3.0a0 preview

This alpha integrates the current nanoribbons development work for Python 3.12
and AiiDA 2.8. It does not replace the stable v2.2.0 release or merge the
compatibility work into `main`.

## Included work

- [#85](https://github.com/nanotech-empa/aiidalab-empa-nanoribbons/pull/85):
  AiiDA 2.8 support, explicit data entry points, installed pseudo-family selection,
  and viewer compatibility with migrated Dict-like output nodes.
- [#86](https://github.com/nanotech-empa/aiidalab-empa-nanoribbons/pull/86):
  combined bands/PDOS Igor export, both spin channels, selected-atom projections,
  and a reusable combined figure for ELN previews.
- [#87](https://github.com/nanotech-empa/aiidalab-empa-nanoribbons/pull/87):
  optional shared openBIS structure importer and preservation of structure origin.
- SVG download from the interactive bands viewer and the combined bands/DOS/PDOS
  plot. PNG, PDF, TXT and Igor downloads remain available. The combined SVG keeps
  labels editable and curves vector-based; it is not an embedded screenshot.
- Importer detection also tolerates an older `aiidalab-eln` installation that
  does not yet expose `OpenbisStructureImporterWidget`.
- Generated numbered notebook copies are excluded from Git.

## Installation

Use a Python >=3.12 AiiDAlab environment with AiiDA >=2.8,<3 and ipywidgets 8.
For a clean instance:

```bash
aiidalab install --yes "aiidalab-empa-nanoribbons@git+https://github.com/nanotech-empa/aiidalab-empa-nanoribbons.git@v2.3.0a0"
```

In a development instance, first check the existing application checkout and
preserve local changes. Restart notebook kernels after installation.
AiiDAlab skips a direct-URL install when the app directory already exists, even
when the requested Git revision differs. Only after preserving that checkout,
add `--force` to the command above to replace it. Use `--dry-run --yes --force`
first to inspect the exact target and installation path without modifying it.

The shared widgets dependency is pinned to the updated
[widgets-base PR #820](https://github.com/aiidalab/aiidalab-widgets-base/pull/820),
commit `f2773b82017d503c40751f9c082f941eef83ee33`, matching Surfaces v2.0.0a2.
Compared with the previous `c40ae9d17584c41bf0c2dfc450949f11bc5abffb` pin,
this changes only four test assertions: runtime code is identical. Aligning
the direct reference avoids conflicting widgets URLs when installing both apps.
No other dependency constraints have been changed for this alpha.

The released `aiida-nanotech-empa` backend already contains the nanoribbon
workflow's `InstalledCode` support; no additional unreleased backend is required.
The nanoribbon workflow in v1.1.1 and the Surfaces v2.0.0a2 backend pin
`45c4307a9db39ac5debc7a2b87f0ad869a425c17` is identical. If using both apps,
retain the Surfaces backend pin for its additional CP2K features.

### Optional openBIS integration

For the shared atomistic-model/molecular-concept importer, install the tested
[aiidalab-eln PR #93](https://github.com/aiidalab/aiidalab-eln/pull/93) revision:

```bash
python -m pip install "aiidalab-eln @ git+https://github.com/aiidalab/aiidalab-eln.git@01e977a67035dd39741b73e3458a919ddfe884fd"
```

Configure openBIS access separately. Without the optional importer, submission
still opens normally and the openBIS tab is omitted. This optional package is
not the separate openBIS simulation-export application.

### Calculation prerequisites

Configure `pw.x`, `pp.x`, and `projwfc.x` codes on the intended computer and install
a suitable pseudopotential family. The notebook suggests the following when its
default family is missing; installation is an explicit administrator/user action:

```bash
aiida-pseudo install sssp --functional PBE --version 1.3 -p precision
```

The submit form selects an installed family and blocks submission when none is
available. No calculations are submitted by the release validation itself.

## Validation and scope

The automated suite checks combined vector export with one/two spin channels,
with/without selected-atom projections, PNG/PDF signatures, the interactive
bands SVG callback, Igor data export, notebook compilation, and optional-importer
absence/older/current cases. It does not execute submission notebook cells.

Run from a source checkout:

```bash
python -m pytest -q tests
python -m black --check nanoribbon tests
python -m flake8 nanoribbon tests
python -m build --no-isolation
```

Browser download interaction and representative end-to-end calculations remain
part of alpha acceptance testing. In particular, the bqplot SVG button uses its
native frontend export API; the unit test verifies the correct export message.
Scientific inputs, workflow calculations, energy references and existing AiiDA
nodes are not changed by this release.

### Local validation, 2026-09-13

- Python 3.12.11, AiiDA 2.8.0, AiiDAlab 26.5.2, ipywidgets 8.1.8,
  bqplot 0.13.1 and the widgets-base revision above.
- 25 automated tests passed; repository-wide Black, isort and Flake8 passed.
- Wheel and source-distribution builds passed. Install the AiiDAlab application
  from the Git tag, not from the Python-module wheel alone.
- Read-only checks against three archived workflows opened both the detailed
  and PDOS viewers, produced vector SVGs with/without atomic projections, and
  exported combined Igor data. These include a two-spin migrated workflow, a
  workflow without cell optimization and a non-spin-polarized workflow.
- Submission structure-manager initialization and `InstalledCode` input-port
  acceptance passed. No new calculation was submitted and existing workflow
  extras were verified unchanged.
- `pip check` passed in the development instance.

Known validation limitations: local mypy reports 16 `import-untyped` diagnostics
for third-party packages without typing metadata; it is not a clean mypy run.
The existing full pre-commit bootstrap has a known aarch64 `setup-cfg-fmt` /
`ukkonen` native-build failure (also recorded in #86/#87); this release uses
direct formatting/lint checks instead of retrying that unrelated compilation.
Existing AiiDA/bqplot/SciPy deprecation warnings remain visible.
