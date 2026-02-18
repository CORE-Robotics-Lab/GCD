from .odis_agent import ODISAgent
from .hissd_agent import HISSDAgent

REGISTRY = {}

REGISTRY["mt_odis"] = ODISAgent
REGISTRY["mt_hissd"] = HISSDAgent
