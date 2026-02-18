REGISTRY = {}

# normal learner
from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .multi_task.hissd_learner import HISSDLearner

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["hissd_learner"] = HISSDLearner
REGISTRY["qtran_learner"] = QTranLearner
