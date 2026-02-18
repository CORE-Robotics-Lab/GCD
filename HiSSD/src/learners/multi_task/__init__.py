from .odis_learner import ODISLearner
from .hissd_learner import HISSDLearner

REGISTRY = {}

REGISTRY["odis_learner"] = ODISLearner
REGISTRY["hissd_learner"] = HISSDLearner
