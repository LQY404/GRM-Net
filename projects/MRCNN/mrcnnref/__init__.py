# from .dataset_mapper import RefcocoDatasetMapper
from .config import add_mrcnnref_config
from .ref_evaluator import RefcocoEvaluator
from .mrcnnref import MRCNNRef
# from .mrgcn import MGCN
# from .mrgcnv2 import MGCN
# from .mrgcnv3 import MGCN
from .mrgcnv4 import MGCN
from .gcnref import GCNRef
from .datamapper_iepref import IEPDatasetMapperWithBasis
from .data_mapper_sketch import SketchDatasetMapper
# from .data_mapper_sketch_new import SketchDatasetMapper
from .datamapper_phrasecut import PhraseCutDatasetMapper
from .resnet_sketch import build_resnet_sketch_fpn_backbone
from .data_mapper_refcoco_single import RefcocoSingleDatasetMapper

# from . import register_refcoco
# from . import register_iepref
from . import register_sketch
from . import register_phrasecut
from . import register_refcoco_single