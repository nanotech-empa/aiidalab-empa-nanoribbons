import ast
import json
from pathlib import Path
from types import SimpleNamespace

import ipywidgets as ipw
from aiida import orm
from aiida_pseudo.groups.family import PseudoPotentialFamily

pytest_plugins = ["aiida.tools.pytest_fixtures"]


def test_installed_family_selector_and_missing_family_hint(aiida_profile_clean):
    path = Path(__file__).resolve().parents[1] / "submit.ipynb"
    source = next(
        "".join(cell["source"])
        for cell in json.loads(path.read_text())["cells"]
        if "def installed_pseudo_families" in "".join(cell["source"])
    )
    names = {
        "PREFERRED_PSEUDO_FAMILY",
        "PSEUDO_INSTALL_HINT",
        "pseudo_family_dropdown",
        "pseudo_info",
        "pseudo_refresh_btn",
    }
    tree = ast.parse(source)
    tree.body = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.FunctionDef)
            or isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id in names
                for target in node.targets
            )
        )
    ]
    button = ipw.Button()
    namespace = {
        "orm": orm,
        "ipw": ipw,
        "PseudoPotentialFamily": PseudoPotentialFamily,
        "style": {},
        "layout": {},
        "btn_submit": SimpleNamespace(btn_submit=button),
    }
    exec(compile(tree, str(path), "exec"), namespace)
    try:
        namespace["refresh_pseudo_families"]()
        assert button.disabled
        assert "aiida-pseudo install sssp" in namespace["pseudo_info"].value
        # Empty test families exercise selection only, not scientific submission.
        PseudoPotentialFamily(label="custom").store()
        preferred = namespace["PREFERRED_PSEUDO_FAMILY"]
        PseudoPotentialFamily(label=preferred).store()
        namespace["refresh_pseudo_families"]()
        assert namespace["pseudo_family_dropdown"].value == preferred
        assert not button.disabled
        namespace["pseudo_family_dropdown"].value = "custom"
        assert namespace["selected_pseudo_family_exists"]()
    finally:
        for name in ("pseudo_family_dropdown", "pseudo_info", "pseudo_refresh_btn"):
            namespace[name].close()
        button.close()
