from types import SimpleNamespace

import ipywidgets as ipw
import pytest
import traitlets
from ase import Atoms

from nanoribbon.assistant import create_panel


def test_bindings_edit_the_real_code_selector_and_ignore_unrelated_state(monkeypatch):
    forms = pytest.importorskip("aiidalab_ai_assistant.forms")

    monkeypatch.setattr(forms, "panel", lambda adapter: adapter)
    dropdown = ipw.Dropdown(options=[("None", None), ("Configured code", "code-uuid")])
    namespace = {
        "pw_code_dropdown": SimpleNamespace(value=None, code_select_dropdown=dropdown),
        "protocol": ipw.Dropdown(options=["standard", "debug"]),
        "credentials": ipw.Password(value="never-in-context"),
        "node": object(),
    }
    adapter = create_panel("submit.ipynb", namespace)
    state = adapter.context()
    assert "never-in-context" not in str(state)
    assert state["controls"]["pw_code_dropdown"]["options"] == [None, "code-uuid"]
    adapter.update(state["revision"], {"pw_code_dropdown": "code-uuid"})
    assert dropdown.value == "code-uuid"
    assert not adapter._actions


def test_chat_replicates_the_actual_ribbon_and_reviews_submission(monkeypatch):
    forms = pytest.importorskip("aiidalab_ai_assistant.forms")
    tool_registry = pytest.importorskip("aiidalab_ai_assistant.tools").ToolRegistry

    from nanoribbon.editors import NanoribbonReplicateEditor

    class Selector(ipw.VBox):
        structure = traitlets.Instance(Atoms, allow_none=True)

    selector = Selector(
        structure=Atoms(
            "C2",
            positions=[[0, 0, 0], [1.42, 0, 0]],
            cell=[2.84, 10, 10],
            pbc=[True, False, False],
        )
    )
    selector.viewer = SimpleNamespace(selection=[])
    editor = NanoribbonReplicateEditor()
    link = traitlets.link((selector, "structure"), (editor, "structure"))
    selector.assistant_editors = (editor,)
    selector.assistant_importers = {}
    submit = ipw.Button(description="Submit sentinel")
    calls = []
    submit.on_click(lambda _: calls.append(True))
    monkeypatch.setattr(forms, "panel", lambda adapter: adapter)
    adapter = create_panel(
        "submit.ipynb", {"structure_selector": selector, "btn_submit": submit}
    )
    registry = tool_registry()
    adapter.register(registry)
    state = adapter.context()
    nx = next(name for name, item in state["controls"].items() if item["label"] == "nx")
    result = registry.call(
        "app_update_draft", {"fields": {nx: 3}, "expected_revision": state["revision"]}
    )
    assert "error" not in result
    state = adapter.context()
    replicate = next(
        name
        for name, item in state["draft_actions"].items()
        if item["label"] == "Replicate"
    )
    result = registry.call(
        "app_prepare_action",
        {"action": replicate, "expected_revision": state["revision"]},
    )
    assert "error" not in result
    assert len(selector.structure) == 6
    assert abs(selector.structure.cell[0, 0] - 8.52) < 1e-10
    args = {"action": "btn_submit", "expected_revision": adapter.context()["revision"]}
    registry.call("app_review_action", args)
    pending = registry.call("app_execute_action", args)
    assert not calls
    registry.approve(pending["action_id"])
    assert calls == [True]
    link.unlink()
    editor.close()
    selector.close()
