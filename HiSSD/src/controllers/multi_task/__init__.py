from .mt_odis_controller import ODISMAC
from .mt_hissd_controller import HISSDSMAC


REGISTRY = {}

REGISTRY["mt_odis_mac"] = ODISMAC
REGISTRY["mt_hissd_mac"] = HISSDSMAC
