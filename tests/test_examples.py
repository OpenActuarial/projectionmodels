import runpy
from pathlib import Path

import pandas as pd
import pytest


EXAMPLE_DIR = Path(__file__).parents[1] / "examples"
EXAMPLE_FILES = sorted(EXAMPLE_DIR.glob("*.py"))


@pytest.mark.parametrize("path", EXAMPLE_FILES, ids=lambda path: path.name)
def test_examples_expose_and_execute_run_example(path):
    namespace = runpy.run_path(str(path), run_name="projectionmodels_example")
    assert "run_example" in namespace
    output = namespace["run_example"]()
    assert isinstance(output, dict)
    assert output
    assert any(
        isinstance(value, pd.DataFrame) or hasattr(value, "to_frame")
        for value in output.values()
    )


@pytest.mark.parametrize("path", EXAMPLE_FILES, ids=lambda path: path.name)
def test_examples_execute_as_scripts(path, capsys):
    runpy.run_path(str(path), run_name="__main__")
    assert capsys.readouterr().out.strip()
