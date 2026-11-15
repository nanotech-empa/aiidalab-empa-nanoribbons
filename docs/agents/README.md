# Helping a user in Nanoribbons

Read `app_get_state` to identify submit/search/show/PDOS/compare context. The
chat uses the shared providers, conversations, AiiDA inspection and openBIS tools.
It edits only the declared visible fields using a fresh revision. This app uses
QE executables through its own workflow; the separate aiidalab-qe app has no
chat integration in this change.

When asked to fill a form, perform the draft edits and read back the result.
Find nested controls by their `controls.label` and option labels. Use current
field keys and revisions, especially after geometry changes. `draft_actions`
lists the existing edit/apply controls, invoked through `app_prepare_action`.
`execution_actions` lists submission controls; only `app_execute_action` after
`app_review_action` and the user's visible execution button runs one.

Use `app_search_structures` and `app_load_structure` for an intended AiiDA
StructureData/CifData UUID. Use `app_import_structure` for supplied standard text
structure files or CDXML. A normal GUI CDXML upload is visible to the chat too:
inspect `context.cdxml`, select `cdxml.structure_selector`, set the intended repeat
count/options, and call `app_cdxml_action(action="create_model")`. Confirm its
messages, formula and cell; do not confuse upload with completed model creation.
The Supercell editor's nx control and Replicate draft action are exposed. Review
the initial repeat and final cell to avoid replicating twice. Atom selections for
the geometry editor use one-based indices through `app_select_atoms`.

## Prepare and submit

In submit.ipynb import/build the intended ribbon and inspect replication, cell,
periodicity and tags. Review the up/down atom lists, total charge, cell
optimization choice and calculation label. Atom lists are the GUI's one-based
ranges; preserve their meaning. Initial spin tags are guesses, not a ground-state
claim. Do not silently substitute a different magnetic/structural model.

Choose compatible PW, PP and PROJWFC Codes and an installed pseudopotential
family. Missing pseudos/Codes are configuration issues; do not install or replace
them merely to suppress a message. Check maximum nodes, memory per node and wall
hours against the selected computer. Inspect the full form and use its existing
Submit button or the reviewed chat execution action. Record the AiiDA
workflow UUID and use actual reports/status to follow progress.

## Interpret and export

Open search.ipynb, show.ipynb, view_geometry.ipynb or pdos.ipynb for the selected
result, and compare.ipynb for compatible stored runs. Check optimization/SCF
status, units, energy reference, spin channels and k sampling. A browser plot is
not proof of numerical convergence. Compare like structures and conventions;
do not alter normalization or spin assumptions to force agreement.

Use the openBIS structure importer for the intended material, and shared typed
searches/relationship queries for existing laboratory evidence. Duplicate names
need explicit identifiers. For publication use
`openbis_open_gui(workflow="export_simulations")`, select the saved workflow UUID
in the existing common form, review project/collection/result cards, then
**Export**. Navigation itself performs no upload. Unsupported data must be
reported; do not create a new object type or invent missing scientific values.
Archive import remains in the existing Import simulations GUI.
