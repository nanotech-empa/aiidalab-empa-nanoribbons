"""Explicit, read-only access to original QE orbital cubes."""

import hashlib
import posixpath
import re
import tempfile
from pathlib import Path

from aiida import orm
from aiida_quantumespresso.calculations.pp import PpCalculation

from .data import ArraySnapshot


def read_original_orbital(identifier, label):
    """Reparse one retained original without changing any stored result.

    ``identifier`` may name a stored orbital array or its PP calculation.
    Remote access is explicit: callers select this source instead of the normal
    stored-array path. Only plain copied arrays and provenance leave this call.
    """
    if not re.fullmatch(r"K\d{3,}_B\d{3,}", label):
        message = "Select a single exported orbital first."
        raise ValueError(message)
    node = orm.load_node(identifier)
    calculation = node if isinstance(node, orm.CalcJobNode) else node.creator
    if not isinstance(calculation, orm.CalcJobNode):
        message = "The original PP calculation is not available."
        # The caller supplied a valid UUID string but selected unsuitable data.
        raise ValueError(message)  # noqa: TRY004
    filename = f"{PpCalculation._FILPLOT}_{label}{PpCalculation._FILEOUT}"
    retrieved = calculation.outputs.retrieved
    if filename in retrieved.base.repository.list_object_names():
        with retrieved.base.repository.open(filename, mode="rb") as handle:
            content = handle.read()
        source = "retrieved"
    else:
        remote = calculation.outputs.remote_folder
        with tempfile.TemporaryDirectory(prefix="nanoribbon-cube-") as directory:
            destination = Path(directory) / filename
            with remote.computer.get_transport() as transport:
                transport.getfile(
                    posixpath.join(remote.get_remote_path(), filename),
                    str(destination),
                )
            content = destination.read_bytes()
        source = "original work directory"

    # Use the maintained backend parser, including its grid-aligned crop.
    from aiida_nanotech_empa.parsers.pp_parser import PpParser

    units = (
        node.get_array("data_units") if isinstance(node, orm.ArrayData) else "e/bohr^3"
    )
    snapshot = ArraySnapshot.from_data(PpParser.parse_gaussian(content.decode(), units))
    return snapshot, {
        "calculation_uuid": calculation.uuid,
        "filename": filename,
        "source": source,
        "sha256": hashlib.sha256(content).hexdigest(),
    }
