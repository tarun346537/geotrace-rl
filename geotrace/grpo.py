"""NumPy group-relative advantages and masked GRPO objective calculations."""

from typing import cast

import numpy as np
from numpy.typing import ArrayLike

from .constants import CLIP_EPSILON, KL_BETA
from .types import FloatArray


def group_relative_advantages(rewards: ArrayLike, epsilon: float = 1e-8) -> FloatArray:
    """Standardize rewards separately for each prompt (row), population std."""
    rewards = np.asarray(rewards, dtype=float)
    if rewards.ndim != 2 or rewards.shape[0] < 1 or rewards.shape[1] < 2:
        raise ValueError("expected [prompts, group>=2] rewards")
    if not np.all(np.isfinite(rewards)) or not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("rewards must be finite and epsilon must be positive")
    return cast(
        FloatArray,
        (rewards - rewards.mean(axis=1, keepdims=True))
        / (rewards.std(axis=1, keepdims=True) + epsilon),
    )


def clipped_objective(
    logp: ArrayLike,
    old_logp: ArrayLike,
    ref_logp: ArrayLike,
    advantages: ArrayLike,
    mask: ArrayLike,
    *,
    clip_epsilon: float = CLIP_EPSILON,
    kl_beta: float = KL_BETA,
) -> dict[str, float | int]:
    """Return masked per-sequence mean clipped surrogate minus sampled KL.

    Inputs have [group, tokens] shape except advantages [group]. Sequences
    are weighted equally, independent of length. log probabilities belong
    to the same sampled tokens. Policy updates belong to the training backend.
    KL estimator: exp(log(ref)-log(policy))-(log(ref)-log(policy))-1.
    It is nonnegative pointwise but is not the exact distributional KL.
    """
    logp, old, ref = (np.asarray(a, dtype=float) for a in (logp, old_logp, ref_logp))
    advantages, mask = np.asarray(advantages, dtype=float), np.asarray(mask)
    if (
        logp.ndim != 2
        or old.shape != logp.shape
        or ref.shape != logp.shape
        or mask.shape != logp.shape
    ):
        raise ValueError("log probabilities and mask must have matching [group, tokens] shapes")
    if logp.shape[0] == 0 or advantages.shape != (logp.shape[0],):
        raise ValueError("one advantage is required per nonempty sequence group")
    if not np.all((mask == 0) | (mask == 1)) or np.any(mask.sum(axis=1) == 0):
        raise ValueError("each sequence needs at least one unmasked token. mask must be binary")
    if not np.all(np.isfinite(advantages)):
        raise ValueError("advantages must be finite")
    selected = mask.astype(bool)
    for array in (logp, old, ref):
        if not np.all(np.isfinite(array[selected])) or np.any(array[selected] > 0):
            raise ValueError("active token log probabilities must be finite and nonpositive")
    if not 0 < clip_epsilon < 1 or not np.isfinite(kl_beta) or kl_beta < 0:
        raise ValueError("require 0<clip_epsilon<1 and finite kl_beta>=0")
    # Ignore arbitrary padding before arithmetic to avoid NaN * 0 or overflow.
    delta_old = np.where(selected, logp - old, 0.0)
    delta_ref = np.where(selected, ref - logp, 0.0)
    if np.any(np.abs(delta_old) > 80) or np.any(np.abs(delta_ref) > 80):
        raise ValueError("extreme policy ratios. reduce the update or inspect token alignment")
    ratio = np.exp(delta_old)
    sampled_kl = np.expm1(delta_ref) - delta_ref
    surrogate = np.minimum(
        ratio * advantages[:, None],
        np.clip(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages[:, None],
    )
    lengths = mask.sum(axis=1)
    sequence_surrogate = (surrogate * mask).sum(axis=1) / lengths
    sequence_kl = (sampled_kl * mask).sum(axis=1) / lengths
    return {
        "objective": float(np.mean(sequence_surrogate - kl_beta * sequence_kl)),
        "mean_surrogate": float(np.mean(sequence_surrogate)),
        "mean_sampled_kl": float(np.mean(sequence_kl)),
        "active_tokens": int(mask.sum()),
    }
