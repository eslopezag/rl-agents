from typing import Callable

import numpy as np


def cosine_decay_scheduler(
    initial_value: float,
    min_value: float,
    decay_steps: int,
    decay_by_episodes: bool = False,
) -> Callable:
    """
    Returns a cosine decay scheduler function that goes from `initial_value` to
    `min_value` in `decay_steps` (folowing the form of a cosine wave
    half-period) and continues to return `min_value` after that.

    If `decay_by_episodes` is True, the scheduler decays the value every
    episode instad of every step.
    """
    min_fraction = min_value / initial_value

    def scheduler(step: int, episode: int) -> float:
        if decay_by_episodes:
            step = episode

        step = min(step, decay_steps - 1)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * step / (decay_steps - 1)))
        decayed = (1 - min_fraction) * cosine_decay + min_fraction
        return initial_value * decayed

    return scheduler


def exp_decay_scheduler(
    initial_value: float,
    min_value: float,
    decay_steps: int,
    decay_by_episodes: bool = False,
) -> Callable:
    """
    Returns an exponential decay scheduler function that goes from
    `initial_value` to `min_value` in `decay_steps` (folowing the form of a
    cosine wave half-period) and continues to return `min_value` after that.

    If `decay_by_episodes` is True, the scheduler decays the value every
    episode instad of every step.
    """
    decay_rate = (min_value / initial_value) ** (1 / (decay_steps - 1))

    def scheduler(step: int, episode: int) -> float:
        if decay_by_episodes:
            step = episode

        step = min(step, decay_steps - 1)
        return initial_value * decay_rate ** step

    return scheduler


def constant_scheduler(constant_value: float) -> Callable:
    """
    Returns a schedular function that always produces the same value.
    """
    def scheduler(step: int, episode: int) -> float:
        return constant_value

    return scheduler
