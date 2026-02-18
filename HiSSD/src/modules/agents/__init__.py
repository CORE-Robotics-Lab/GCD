REGISTRY = {}

# normal agents
from .rnn_agent import RNNAgent
from .multi_task.hissd_agent import HISSDAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["mt_hissd"] = HISSDAgent
