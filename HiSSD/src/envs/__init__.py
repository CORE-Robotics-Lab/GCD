from functools import partial


# from smac.env import MultiAgentEnv, StarCraft2Env
from .gymma import GymmaWrapper

from .multiagentenv import MultiAgentEnv

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


def gymma_fn(env, **kwargs) -> MultiAgentEnv:
    assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
    return env(**kwargs)


REGISTRY = {}
# REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["gymma"] = partial(gymma_fn, env=GymmaWrapper)

from .smac_v2.smacv2_env import SMACv2
REGISTRY["smacv2"] = partial(env_fn, env=SMACv2)
