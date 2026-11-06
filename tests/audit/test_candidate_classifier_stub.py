"""The installed collision stubs expose the native candidate consumer contract."""

import ast
from pathlib import Path


def test_candidate_classifiers_have_the_ordered_row_consumer_signature():
    stub = (
        Path(__file__).resolve().parents[2]
        / "src/tesseract_robotics/tesseract_collision/_tesseract_collision.pyi"
    )
    tree = ast.parse(stub.read_text())
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for name in ("classify_candidate_state_rows", "classify_candidate_state_rows_first"):
        function = functions[name]
        assert [arg.arg for arg in function.args.args] == [
            "solver",
            "manager",
            "names",
            "rows",
            "limits",
        ]
        assert ast.unparse(function.returns) == "list[str]"
        assert all(arg.annotation is not None for arg in function.args.args)
