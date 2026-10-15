[![DOI](https://zenodo.org/badge/188820720.svg)](https://zenodo.org/badge/latestdoi/188820720)

# Materials Cloud - Empa Nanoribbon App

This [Materials Cloud jupyter](https://jupyter.materialscloud.org) app is a GUI for
AiiDA workflows that allows to easily compute and plot in a standardized way
basic DFT properties for graphene-based nanoribbons.

The nanoribbon workflow can be subdivided in three sections:
 * Upload/ modification of Structures
 * Calculation of properties
 * Search / visualization in the database

To get an idea of how it works, please [check out the videos](https://www.youtube.com/playlist?list=PL19kfLn4sO_8O_yQTL6KK0nC2adrrLqmi).

## Uploading a new structure
File formats compatible with [ASE](https://wiki.fysik.dtu.dk/ase/) can be used.
In the example a .mol file for a structure is generated with ChemDraw.

The structure is uploaded, a warning message is provided if similar structures were already computed.

The structure is stored in the database.

The structure created with ChemDraw is properly scaled to conventional C-C sp2 bond lengths.

A unit cell is created identifying two equivalent atoms in the edited (scaled) structure.

## Submission of a calculation

 1. Select from the list of edited structure a structure that has a unit cell defined (it is also possible to directly upload structures with predefined unit cell thus skipping the steps 3-4 of the upload procedure)
 1. Select a computer from the list of installed computers
 1. Select the desired precision in terms of k-point samplig (1 corresponds to 60 k-pt for a unit cell of size 2.4 A)
 1. Submit the calculation

The status of the workflow can be monitored either from the AiiDA status or from the list of submitted workflows.

## For maintainers

To create a new release, clone the repository, install development dependencies with `pip install '.[dev]'`, and then execute `bumpver update --major/--minor/--patch`.
This will:

  1. Create a tagged release with bumped version and push it to the repository.
  2. Trigger a GitHub actions workflow that creates a GitHub release.

Additional notes:

  - Use the `--dry` option to preview the release change.
  - The release tag (e.g. a/b/rc) is determined from the last release.
    Use the `--tag` option to switch the release tag.

## Contact

For inquiries concerning Materials Cloud, please use the [public issue tracker](https://github.com/materialscloud-org/issues).
For inquiries concerning the Empa Nanoribbon App, please contact [carlo.pignedoli@empa.ch](mailto:carlo.pignedoli@empa.ch).

## Software attribution and scientific acknowledgements

Please credit [aiidalab-empa-nanoribbons](https://github.com/nanotech-empa/aiidalab-empa-nanoribbons) and its
[contributors](https://github.com/nanotech-empa/aiidalab-empa-nanoribbons/graphs/contributors),
and identify the release or immutable commit used. Existing author,
contributor and funding acknowledgements remain authoritative.

See [CITATION.cff](CITATION.cff) for the software citation. A project DOI
covers archived releases; identify the exact version or commit separately.
Use a release DOI only for the release it identifies.

Preserve the copyright and permission notices in [LICENSE.txt](LICENSE.txt)
when copying or distributing copies or substantial portions of MIT-licensed
code. Separately licensed components retain their own notices. A name or
repository link does not replace the applicable licence notices.

For research using this app, acknowledge the [AiiDA team](https://github.com/nanotech-empa/.github/blob/main/CITATION_GUIDANCE.md#aiida)
and [AiiDAlab team](https://github.com/nanotech-empa/.github/blob/main/CITATION_GUIDANCE.md#aiidalab), and cite the three AiiDA papers
and both AiiDAlab platform papers listed in that guide.

Also credit aiida-nanotech-empa, aiida-quantumespresso, Quantum ESPRESSO and the methods actually used. This app is distinct from the AiiDAlab Quantum ESPRESSO app.

The [shared credit guide](https://github.com/nanotech-empa/.github/blob/main/CITATION_GUIDANCE.md) keeps software attribution, licence
compliance and scientific citation requests separate. Scientific citation
requests are not additional MIT licence conditions.
