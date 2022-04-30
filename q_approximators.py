from abc import ABC, abstractmethod
from typing import Optional, Union, List, Callable, Tuple

import numpy as np
from numpy import typing as npt
import tensorflow as tf
import gym

from .experience_buffers import ExperienceBufferBase, OrderedExperienceBuffer


class QApproximatorBase(ABC):
    @abstractmethod
    def update(
        self,
        last_state: Union[int, float, npt.NDArray[np.float64]],
        last_action: int,
        reward: float,
        new_state: Union[int, float, npt.NDArray[np.float64]],
        new_action: int,
        target_fn: Callable,
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

    def update(
        self,
        last_state: Union[int, float, npt.NDArray[np.float64]],
        last_action: int,
        reward: float,
        new_state: Union[int, float, npt.NDArray[np.float64]],
        new_action: int,
        target_fn: Callable,
    ) -> None:
        target = target_fn(reward, new_state, new_action)
        self[last_state, last_action] += self.step_size * (
            target - self[last_state, last_action]
        )


class GradQApproximator(QApproximatorBase):
    def __init__(
        self,
        model: Callable,
        optimizer: tf.keras.optimizers.Optimizer,
        state_dim: int,
        buffer_size: int,
        buffer_class: type = OrderedExperienceBuffer,
        scheduler: Optional[Callable] = None,
        is_terminal_fn: Optional[Callable] = None,
        *,
        env: Optional[gym.Env] = None,
        n_actions: Optional[int] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer

        if not ExperienceBufferBase.__subclasscheck__(buffer_class):
            raise ValueError(
                'The `buffer_class` parameter must be a subclass of '
                f'{ExperienceBufferBase}.'
            )

        self.buffer = buffer_class(buffer_size, state_dim)
        self.scheduler = scheduler
        self.is_terminal_fn = is_terminal_fn

        if env is not None:
            self.n_actions = env.action_space.n
        elif n_actions is not None:
            self.n_actions = n_actions
        else:
            raise ValueError(
                'Either `env` or `n_actions` must be specified.'
            )

        self.shape = (np.inf, self.n_actions)

        self.masks = np.eye(self.n_actions, dtype=bool)

    def __getitem__(
        self,
        state_action_tuple: Union[
            Tuple[
                Union[int, float, npt.NDArray[np.float64]],
                Union[int, slice]
            ],
            Union[int, float, npt.NDArray[np.float64]]
        ],
    ) -> float:
        if isinstance(state_action_tuple, tuple):
            state, action = state_action_tuple
        else:
            state = state_action_tuple
            action = slice(None)

        if isinstance(state, (int, float)):
            return self.model.predict(
                np.array(state, dtype=np.float64)[None, None]
            ).squeeze()[action]
        else:
            return self.model.predict(state[None]).squeeze()[action]

    def _calculate_loss(self, target_fn: Callable) -> tf.Tensor:
        prev_Qs = tf.boolean_mask(
            self.model(self.buffer['prev_states']),
            self.masks[self.buffer['prev_actions']]
        )

        target = target_fn(
            self.buffer['rewards'],
            self.buffer['next_states'],
            self.buffer['next_actions'],
            self.buffer['next_is_terminal'],
        )

        loss = tf.reduce_mean(tf.square(target - prev_Qs))

        return loss

    def update(
        self,
        last_state: Union[int, float, npt.NDArray[np.float64]],
        last_action: int,
        reward: float,
        new_state: Union[int, float, npt.NDArray[np.float64]],
        new_action: int,
        target_fn: Callable,
    ) -> None:
        self.buffer.add(
            last_state,
            last_action,
            reward,
            new_state,
            new_action,
            self.is_terminal_fn(new_state),
        )

        with tf.GradientTape() as tape:
            loss = self._calculate_loss(target_fn)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables)
        )

    def run_scheduler(self, step: int, episode: int) -> None:
        """
        Runs the scheduler of the approximator to vary its step size during
        training.
        """
        if self.scheduler:
            self.optimizer.learning_rate = self.scheduler(step, episode)
