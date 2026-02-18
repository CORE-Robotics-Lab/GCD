REGISTRY = {}

from .sc2_decomposer import SC2Decomposer
from .smacv2_decomposer import SMACv2Decomposer

REGISTRY["sc2_decomposer"] = SC2Decomposer
REGISTRY["smacv2_decomposer"] = SMACv2Decomposer   

# from .gymma_decomposer import GYMMADecomposer

# REGISTRY["gymma_decomposer"] = GYMMADecomposer
