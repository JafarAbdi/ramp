import runpy
import pytest
import pathlib

FILE_PATH = pathlib.Path(__file__).parent
EXAMPLES = (FILE_PATH / ".." / "examples").glob("*.py")


@pytest.mark.parametrize("example", EXAMPLES)
def test_example(example: pathlib.Path):
    if example.name in ["mj_ik.py", "mj_interactive.py"]:
        pytest.skip(f"{example} is not ported example")
    runpy.run_path(example)
