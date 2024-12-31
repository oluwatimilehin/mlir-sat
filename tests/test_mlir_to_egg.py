
import difflib
import logging

from pathlib import Path


from converter import Converter
from test_util import TestUtil
from xdsl.parser import Parser as IRParser




logger = logging.getLogger(__name__)


def test_mlir_to_egg():
    current_dir = Path(__file__).parent
    eggs_path = f"{current_dir}/data/eggs"
    mlir_path = f"{current_dir}/data/mlir"

    mlir_files = [f for f in Path(mlir_path).iterdir()]

    for mlir_file in mlir_files:
        mlir_file_name = Path(mlir_file).stem
        logger.info(f"Processing mlir: {mlir_file_name}")

        with open(mlir_file) as f:
            mlir_parser = IRParser(
                TestUtil.context(), f.read(), name=f"{mlir_file_name}"
            )
            mlir = mlir_parser.parse_module()
            actual_egg = Converter.to_egglog(mlir)

            egg_file = f"{eggs_path}/{mlir_file_name}.egg"
            expected_egg_expr = Path(egg_file).read_text()

            diffs = []
            for i, s in enumerate(
                difflib.ndiff(
                   str(actual_egg).strip(), expected_egg_expr.strip()
                )
            ):
                match s[0]:
                    case "-":
                        diffs.append(f"Delete {s[-1]} from position {i}")
                    case "+":
                        diffs.append(f"Add {s[-1]} to position {i}")
                    case _:
                        continue

            assert not diffs

            # assert str(actual_egg).strip() == expected_egg_expr.strip()
