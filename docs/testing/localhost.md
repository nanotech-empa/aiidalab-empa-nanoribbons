# Nanoribbons — local manual GUI qualification

This guide covers current `nanotech-empa/aiidalab-empa-nanoribbons` main and
its actual `nanotech_empa.nanoribbon` QE workflow. The modern ELN/openBIS importer
is included in the app; live-server writes, the frozen legacy line and XPS
scientific calculations remain outside these localhost cases.

Start only after the unified source/installation readiness report, PASS preflight
and the user's go-ahead. The operator uses the actual browser; CLI/backend-only
runs do not qualify these cases. Save canonical/installed/imported revisions,
dependency versions, source/manifest fingerprints, Code/Computer and result UUIDs,
controls/native inputs, timing/resources/files, exports and reopening routes.
Source changes invalidate affected cases. Keep functional, convergence and
physical verdicts separate. PASS meets the frozen criterion; FAIL contradicts
it; BLOCKED names a prerequisite; SKIPPED names an explicit exclusion.

## Actual controls and resource admission

This app uses QE, not the Surfaces CP2K protocol form. It has no **Standard** or
**Low accuracy** selector. The submit notebook fixes `precision=1.0`; the backend
uses its existing stage-specific sampling factors. Do not invent an accuracy
selector, edit the notebook for a test, or describe this recipe as CP2K Standard.
The campaign's Standard/Low restriction still applies to its CP2K cases.

Use one native job at a time and localhost pw.x, pp.x and projwfc.x on the same
existing Computer. Set **Max # nodes = 1**, **Max memory/node (GB) = 32** and
**Max hours = 1**, the smallest positive integer-hour GUI value. The memory field
is a resource-planning input, not a hard memory limit. Observe actual available
RAM and resident usage. MPI ranks come from the Computer default; there is no
per-job rank selector here. Require an already configured default of 2, 4 or 8,
one core/thread per rank, and inspect every native allocation. The aligned
campaign Computer has 8; projwfc caps its ranks at 4. Do not modify shared Computer
defaults or select remote compute to make the case fit.

**Native walltimes differ by stage.** With **Max hours = 1**, the current
source sets SCF 600 s, bands 900 s, bands_lowres 300 s, Hartree pp 1800 s,
projwfc 3600 s and orbital pp a cell-volume-derived limit. These are cooperative
ceilings, not predicted runtimes. Longer walltimes are already authorized for
this serial localhost functional campaign; they do not require another approval.

Observe progress, actual available RAM, resident usage, disk space and file
growth every 30 seconds. Record the controls and actual native ceilings before
submission, and preserve observed quantities and native diagnostics. Stop new
testing and report a serious fault under the existing campaign instructions.
Use the agreed supported exact-root stop route if required; preserve logs and
checkpoints. Native termination or Finished0 alone is not scientific PASS.

## N00 — Entry, import, cell and replication (no submission)

1. Open **Submit calculation** (`submit.ipynb`). Require **From computer**,
   **AiiDA database** and **CDXML to GNR**, with **Main** and **Supercell** editors.
   Opening/selecting the app starts no calculation. These cases use local data;
   they do not exercise production openBIS server writes.
2. Upload the provisional fixture
   [4agnr-c8h4-xyz.extxyz](fixtures/4agnr-c8h4-xyz.extxyz). It is a hydrogen-
   passivated four-dimer-line armchair ribbon, 12 atoms C8H4, X period 4.26 Å and
   transverse lengths 15 Å. The construction record is alongside the fixture.
   Confirm atom order, C/H identities, planar geometry and actual cell/PBC.
3. This is structurally a one-dimensional ribbon but the chosen calculation is
   a **3D periodic array with transverse vacuum**, PBC XYZ. The QE recipe uses
   full periodic electrostatics; no isolated-system or1D-Coulomb setting is
   applied. Never silently replace the explicit cell or infer electrostatics
   from structural connectivity alone.
4. In **Supercell**, set **nx = 2**, click **Replicate** once and require 24 atoms,
   doubled X length and unchanged transverse lengths. Another click repeats
   the current structure rather than resetting it. Restore the 12 atom fixture
   by a fresh import/session and verify coordinates before native preparation.
   In Main exercise selection/move/undo and restore the exact fixture.
5. Inspect supported viewer representations and structure export if offered.
   Check downloaded coordinates/cell/PBC. The manager is configured storable=False;
   do not claim a separate Store action exists. Official widgets-base3.0.1's
   known same-node Load-after-edits limitation remains pending upstream PR832.
   A fresh import/reopen is not evidence that this specific limitation passed.
6. With an optional tiny CDXML fixture, inspect scale, carbon/hydrogen mapping,
   bonds and periodic-axis construction before accepting it. Missing fixture
   or unsupported chemical content is BLOCKED, not covered by the XYZ upload.

## N01 — Calculation draft and guards (no submission)

1. On the restored12atom fixture choose the three localhost **PW code**,
   **PP code**, **PROJWFC code** entries. Confirm the same Computer and reviewed
   binary/pseudo identities. Set **Pseudo family** to an existing compatible PBE
   family; inspect **Refresh pseudos** and missing-family feedback. Do not install
   pseudos/dependencies or change the existing family during a test.
2. Set unique **Calculation Name**, **Total charge = 0**, empty **IDs atoms spin
   U/D**, and **Opt. cell = OFF** for the smallest frozen-geometry path. Capture
   the resource controls and effective allocation described above. Blank spin
   lists must keep the unpolarized C/H model. Do not reinterpret numeric kind
   suffixes as new chemical elements.
3. Inspect missing structure/code/family, short calculation name and invalid spin
   drafts without submitting them. Record actual validation and absent help.
   An invalid spin edit must not rename or alter an unrelated stored structure.
   The existing spin-mismatch callback can create a new tagged structure and
   rename its selected source; exercise that path only on an owned disposable
   fixture if separately selected, preserving its original record/evidence.
4. There is no durable Review token in this form. The submit button being disabled
   during a callback is not durable duplicate protection. Do not launch a second
   real job to probe this gap. Editing a draft must not make old results appear
   to describe the new input.

## N02 — Tiny frozen ribbon, one GUI submission

Needs N00/N01 and the single campaign test-start go-ahead. This covers the
described tiny case under the existing serial localhost resource and longer-
walltime authorization. It uses several serial QE/pp steps; no benchmark has
been performed. Cell optimization, magnetic variants and denser/wider ribbons
are separate extensions, not prerequisites for this first case.

1. Recheck the final draft and press Submit once. Record the root UUID and every
   child UUID/label. No native submission from a terminal. Require one active
   native job across the entire campaign.
2. With Opt. cell OFF require no cell_opt1/cell_opt2 job and unchanged stored
   working geometry/cell. Expected sequence: scf → export_hartree → bands →
   export_pdos → bands_lowres → serial export_orbitals jobs. For the unpolarized
   model export_spinden is skipped. Up to 8 frontier bands are exported, limited
   by available states; audit the actual selected band range and child count.
3. Audit PBE pseudopotentials,50Ry wavefunction/400Ry density cutoffs, charge 0,
   one spin component, 36 valence electrons assuming C/H pseudopotentials with 4/1 valence electrons, and explicit XYZ
   cell. The current unoptimized fixture generates 52 SCF, 70 band-path and 12 low-
   resolution sampling points before symmetry reduction, subject to the stored
   backend inputs. Check the actual grids and primitive reciprocal-axis units.
   Γ→X runs along the first reciprocal axis from 0 to 0.5. No primitive-cell or
   dimension change is intended.
4. Require actual SCF convergence, finite consistent native/parsed energies,
   occupation/electron count and eigenvalues. The recipe sets
   `scf_must_converge=False`; a successful process/native footer alone therefore
   cannot establish convergence. Bands must include occupied/frontier states
   and preserve k/spin/band identities. This small ribbon is expected to be
   gapped, but no converged numerical gap is asserted by this fixture.
5. Audit pp/projwfc outputs, wavefunction projections, potential grids and orbital
   density arrays for finite values, geometry/origin/grid consistency and units.
   Use native/parsed unit tags; do not apply an ORCA Hartree conversion to an
   already-eV QE array. No charge-difference cube or orbital phase is implied
   by an exported |ψ|² map. Do not enforce a complete-basis PDOS sum rule on
   truncated atomic projections or a restricted energy window.
6. Record observed resources, elapsed times, maxima as sampled values and actual
   work/retrieved growth. Failure/timeout retains a partial functional verdict;
   do not call it scientific PASS. Record actual ceilings and any changes.

## N03 — Search, result views and durable reopening

No native jobs. **Search preprocesses eligible sealed NanoribbonWorkChain nodes
and writes derived extras before applying table filters.** The independent
operator/auditor must identify eligible historical records first. If unrelated
protected records would be modified, block this Search branch pending a scoped
GUI solution; do not bulk-reprocess them or manually mark them processed. Only
new owned test metadata is within this case. Preserve all original science.

1. Open **Search database** (`search.ipynb`). Use actual **PKs**, **Formulas**,
   **Calculation Name**, gap/frontier/Fermi/magnetization filters and **Search**.
   Match the recorded root UUID/PK; do not invent a UUID input absent from UI.
2. Follow **Show** (`show.ipynb?id=<root-PK>`) and **PDOS**
   (`pdos.ipynb?id=<root-PK>`). Compare geometry, electron/spin information,
   vacuum/energy reference, native bands and available orbital densities with
   N02. Changing Band, k-point, spin or plot3D must update the selected data.
   Optional effective-mass fits need sufficient locally quadratic sampling;
   a displayed fit alone does not validate effective-mass physics.
   **Orbital source** normally reads **Stored results**. To inspect retained
   original orbital cubes with the current backend parser, explicitly choose
   **Original cube files**. The same source is used for 2D/3D orbitals and LDOS;
   stored arrays are never rewritten and no calculation is submitted. The
   original PP files must remain accessible on their calculation computer.
   Check the displayed PP UUID, filename and SHA256, and verify geometry/grid
   alignment against the original cube. Switching sources clears the previous
   LDOS map and its download; a missing original must show an unavailable
   message rather than silently displaying the stored result. Spin-density
   selection continues to use its separate stored output.
   LDOS uses the existing equally spaced Gamma-X half-path: Gamma and X have
   multiplicity 1 and each interior k-point has multiplicity 2 for its -k
   partner. Gaussian broadening, energy reference and density clamping retain
   their existing definitions. This is not a general arbitrary-k-mesh weighting.
3. In PDOS set **Broadening [eV] = 0.1**, **Methfessel-Paxton order = 0** for a
   Gaussian inspection, choose **Emin/Emax**, **PDOS color**, **Max band width**
   and **Plot**. Select all atoms and C/H subsets; require indices and colors
   to match the displayed bands/projections. Higher Methfessel-Paxton orders
   can have negative tails and are not a positivity error by themselves.
4. Export actual PNG, PDF, SVG, band-text and IGOR links. Parse numeric columns,
   k units, reference shifts, spin/atom selection and energy grids against stored
   arrays at printed precision. SVG must retain vector curves and editable
   labels, without replacing the plot by a raster image. Repeat after changing
   selection/range and require no stale download. Do not claim a ZIP action.
5. Geometry links use `view_geometry.ipynb?uuid=<structure-UUID>`; the structure
   export route is `export_structure.ipynb?uuid=<structure-UUID>`. Check actual
   offered downloads and independently read coordinates/cell/PBC. Close the
   completed view, reopen the same root through Search/fresh URL and compare
   the same data without jobs. Missing outputs must not reuse a previous run.

## N04 — Optional optimization and spin extensions

Draft-only before selection. Opt. cell ON adds two sequential vc-relax stages,
precision factors 0.5 then 1, with only the X cell degree of freedom. This is not a
user-selectable Standard/Low pair. Record the actual controls for any selected optimization case; require real
ionic/cell convergence and unchanged Y/Z vacuum. Spin
variants require owned fixtures, explicit tagged-kind/index mapping, a justified
spin reference and native magnetization checks. Neither branch passes from the
unpolarized frozen fixture. No remote computer or large ribbon is a fallback.

## DOC01 and browser housekeeping

Record help gaps for fixed versus exposed precision, MPI inheritance, planning
memory versus hard limits, actual walltime scaling, repeated-cell boundary
conditions, spin/source-description effects, Search preprocessing, energy units,
projection completeness and exported quantities. Identify maintained app/backend
sources useful to AiiDAlab/in-app assistants; do not claim chatbot retrieval
without evidence. Preserve scientific software/authorship/citation notices.

After each completed manual case record verdict, UUIDs, evidence and reopen
route, then close only that case's Chrome tabs. Keep active tests, pending user
reviews and unrelated tabs. Closing a tab must not stop a calculation.
