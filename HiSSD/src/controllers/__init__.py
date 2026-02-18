REGISTRY = {}

from .basic_controller import BasicMAC
from .multi_task.mt_hissd_controller import HISSDSMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["mt_hissd_mac"] = HISSDSMAC
