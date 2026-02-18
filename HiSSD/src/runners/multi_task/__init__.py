from .episode_runner import EpisodeRunner as MultiTaskEpisodeRunner
from .episode_ada_runner import AdaEpisodeRunner as MultiTaskAdaEpisodeRunner
from .parallel_runner import ParallelRunner as MultiTaskParallelRunner

REGISTRY = {}

REGISTRY["mt_episode"] = MultiTaskEpisodeRunner
REGISTRY["mt_ada_episode"] = MultiTaskAdaEpisodeRunner
REGISTRY["mt_parallel"] = MultiTaskParallelRunner