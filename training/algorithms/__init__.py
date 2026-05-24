"""Algorithm implementations: PPO and FlashSAC.

Both trainers expose the same `train(env, cfg, logger, **kwargs)` surface and
checkpoint the same way, so `train.py` can pick between them by name only.
"""

from training.algorithms.ppo import train_ppo, train_ppo_vec
from training.algorithms.flashsac import train_flashsac_vec

__all__ = ["train_ppo", "train_ppo_vec", "train_flashsac_vec"]
