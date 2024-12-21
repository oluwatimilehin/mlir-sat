from converter import Converter

from test_util import TestUtil

from xdsl.parser import Parser as IRParser

from pathlib import Path
import logging


logger = logging.getLogger(__name__)


def test_egg_to_mlir():
    current_dir = Path(__file__).parent
    eggs_path = f"{current_dir}/data/egg"
    mlir_path = f"{current_dir}/data/mlir"

    egg_files = [f for f in Path(eggs_path).iterdir()]

    for egg_file in egg_files:
        egg_name = Path(egg_file).stem
        logger.info(f"Processing egg: {egg_name}")

        egg_expr = Path(egg_file).read_text()
        actual_mlir = Converter.to_mlir(egg_expr, TestUtil.context())

        expected_mlir_file = f"{mlir_path}/{egg_name}.mlir"
        with open(expected_mlir_file) as f:
            mlir_parser = IRParser(
                TestUtil.context(), f.read(), name=f"{expected_mlir_file}"
            )
            expected_mlir = mlir_parser.parse_module()

            assert actual_mlir.is_structurally_equivalent(expected_mlir)
