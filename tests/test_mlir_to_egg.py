from converter import Converter

from test_util import TestUtil

from xdsl.parser import Parser as IRParser

from pathlib import Path
import logging


logger = logging.getLogger(__name__)


def test_mlir_to_egg():
    current_dir = Path(__file__).parent
    eggs_path = f"{current_dir}/data/egg"
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

            assert str(actual_egg) == expected_egg_expr
