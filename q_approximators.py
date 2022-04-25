from abc import ABC, abstractmethod
from typing import Optional, Union, List, Callable

import numpy as np
from numpy import typing as npt
import gym


class QApproximatorBase(ABC):
    @abstractmethod
    def update(self, state: int, action: int, target: float) -> None:
        pass

    @abstractmethod
    def batch_update(
        self,
        states: npt.NDArray[np.int64],
        actions: npt.NDArray[np.int64],
        targets: npt.NDArray[np.float64],
    ) -> None:
        pass

    def run_scheduler(self, step: int, episode: int) -> None:
        """
        Runs the scheduler of the approximator to vary its step size during
        training.
        """
        if self.scheduler:
            self.step_size = self.scheduler(step, episode)


class TabularQApproximator(np.ndarray, QApproximatorBase):
    """
    Action-value approximator for the tabular case. It is basically an numpy
    array with an `update` method. The class `np.ndarray` was subclassed
    according to https://numpy.org/doc/stable/user/basics.subclassing.html
    """
    def __new__(
        cls,
        input_array: npt.ArrayLike,
        step_size: Optional[float] = None,
        scheduler: Optional[Callable] = None,
        terminal_states: Optional[
            Union[List[int], npt.NDArray[np.int64]]
        ] = None
    ) -> 'TabularQApproximator':
        arr = np.asarray(input_array)

        # Set the action-value estimates of the terminal states to zero:
        if terminal_states:
            arr[terminal_states] = 0.

        obj = super().__new__(
            cls,
            shape=arr.shape,
            dtype=np.float64,
            buffer=arr,
        )

        obj.scheduler = scheduler

        if step_size is None:
            if scheduler is None:
                raise ValueError(
                    'If the `step_size` parameter is not set, then the '
                    'scheduler parameter must be.'
                )

            obj.step_size = scheduler(0, 0)
        else:
            obj.step_size = step_size

        return obj

    def __array_finalize__(self, obj: Optional[npt.NDArray]):
        if obj is None:
            return

        self.step_size = getattr(obj, 'step_size', None)
        self.scheduler = getattr(obj, 'scheduler', None)

        if self.step_size is None:
            if self.scheduler is None:
                raise ValueError(
                    'The object cannot be cast as a `TabularQApproximator`'
                )

            self.step_size = self.scheduler(0, 0)

    @classmethod
    def from_normal_dist(
        cls,
        step_size: Optional[float] = None,
        scheduler: Optional[Callable] = None,
        initial_Q_mean: float = 0.,
        initial_Q_std: float = 0.,
        *,
        env: Optional[gym.Env] = None,
        n_states: Optional[int] = None,
        n_actions: Optional[int] = None,
        terminal_states: Optional[
            Union[List[int], npt.NDArray[np.int64]]
        ] = None
    ) -> 'TabularQApproximator':
        """
        Constructs a `TabularQApproximator` with the given step size,
        initializing the action-value estimates according to a normal
        distribution with mean `initial_Q_mean` and standard deviation
        `initial_Q_std`. The action-value estimates of the terminal states are
        then set to zero.
        """
        if env is not None:
            n_states = env.observation_space.n
            n_actions = env.action_space.n
        else:
            if n_states is None or n_actions is None:
                raise ValueError(
                    'Either `env` or both `n_states` and `n_actions` must be '
                    'specified.'
                )

        return cls(
            np.random.normal(
                loc=initial_Q_mean,
                scale=initial_Q_std,
                size=(n_states, n_actions),
            ),
            step_size,
            scheduler,
            terminal_states,
        )

    def update(self, state: int, action: int, target: float) -> None:
        self[state, action] += self.step_size * (target - self[state, action])

    def batch_update(
        self,
        states: npt.NDArray[np.int64],
        actions: npt.NDArray[np.int64],
        targets: npt.NDArray[np.float64],
    ) -> None:
        raise NotImplementedError(
            '`TabularQApproximator` cannot perform batch updates.'
        )
