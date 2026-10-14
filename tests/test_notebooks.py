"""Validate notebook source without executing submission or changing AiiDA nodes."""

import ast
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
from IPython.core.inputtransformer2 import TransformerManager

APP_ROOT = Path(__file__).parents[1]


def notebook_source(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    transformer = TransformerManager()
    return "\n".join(
        transformer.transform_cell("".join(cell["source"]))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


@pytest.mark.parametrize(
    "path", sorted(APP_ROOT.glob("*.ipynb")), ids=lambda path: path.name
)
def test_notebook_source_compiles(path):
    compile(notebook_source(path), path.name, "exec")


@pytest.mark.parametrize("availability", ("absent", "older", "available"))
def test_optional_openbis_importer(monkeypatch, availability):
    tree = ast.parse(notebook_source(APP_ROOT / "submit.ipynb"))
    symbol = "OpenbisStructureImporterWidget"
    imports = [
        node
        for node in tree.body
        if isinstance(node, ast.Try)
        and any(
            isinstance(child, ast.ImportFrom) and child.module == "aiidalab_eln"
            for child in node.body
        )
    ]
    selections = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Starred)
        and isinstance(node.value, ast.IfExp)
        and any(
            isinstance(child, ast.Name) and child.id == symbol
            for child in ast.walk(node.value)
        )
    ]
    assert len(imports) == len(selections) == 1
    module = ModuleType("aiidalab_eln")
    if availability == "available":
        setattr(module, symbol, lambda **kwargs: kwargs["title"])
    monkeypatch.setitem(
        sys.modules, "aiidalab_eln", None if availability == "absent" else module
    )
    # Execute only the optional import and list entry, never workflow submission.
    namespace = {}
    exec(
        compile(ast.Module(body=imports, type_ignores=[]), "submit.ipynb", "exec"),
        namespace,
    )
    result = eval(
        compile(ast.Expression(selections[0]), "submit.ipynb", "eval"), namespace
    )
    assert result == (["From openBIS"] if availability == "available" else [])
