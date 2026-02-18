from functools import partial
import sys
import os
from argparse import Namespace

from .multiagentenv import MultiAgentEnv

# from .starcraft import StarCraft2Env
# from .myenv import EqualLine,Consensus
from .fire_commander.fire_commander_env import FireCommanderEnv
from .ocean.OceanRealMultiEnv import OceanRealMulti
from .smacv2.smacv2_env import SMACv2

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

# def env_fn(env, **kwargs) -> MultiAgentEnv:
#     return env(Namespace(**kwargs))

REGISTRY = {}
REGISTRY["fc"] = partial(env_fn, env=FireCommanderEnv)
REGISTRY["ocean"] = partial(env_fn, env=OceanRealMulti)
REGISTRY["smacv2"] = partial(env_fn, env=SMACv2)

# REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
# REGISTRY["equal_line"] = partial(env_fn, env=EqualLine)
# REGISTRY["consensus"] = partial(env_fn, env=Consensus)

# if sys.platform == "linux":
#     os.environ.setdefault("SC2PATH", "~/StarCraftII")
