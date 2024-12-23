from eggie.eclasses.base import Region
from eggie.parser import EgglogParser

from mlir.parser import MLIRParser

from xdsl.context import MLContext
from xdsl.dialects.builtin import ModuleOp
from xdsl.parser import Parser as IRParser


class Converter:
    @classmethod
    def to_egglog(cls, module_op: ModuleOp) -> Region:
        return MLIRParser(module_op).parse()

    @classmethod
    def to_mlir(cls, region: Region, context: MLContext) -> ModuleOp:
        egglog_parser = EgglogParser(region)
        region_ast = egglog_parser.parse()
        print(f"region_ast: {region_ast}")
        mlir_parser = IRParser(context, str(region_ast))
        return mlir_parser.parse_module()
