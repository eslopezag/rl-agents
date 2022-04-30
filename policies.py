from abc import ABC, abstractmethod
from typing import Optional, Union, List, Callable
from functools import wraps

import numpy as np
from numpy import typing as npt
import tensorflow as tf
import gym

from .q_approximators import QApproximatorBase, GradQApproximator


StateType = Union[int, float, List[float], npt.NDArray[np.float64]]


class ShapelessQError(ValueError):
    def __init__(self, message: Optional[str] = None) -> None:
        if message is not None:
            super().__init__(message)
        else:
            super().__init__(
                'The `Q` attribute should have a `shape` attribute of its own '
                'pointing to a tuple of the form `(n_states, n_actions)`.'
            )


def multi_argmax(arr: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
    """
    Gets all the indices of the array where its maximum value is reached.
    """
    return np.flatnonzero(arr == np.max(arr))


def random_argmax(arr: npt.NDArray[np.float64]) -> int:
    """
    Gets the index of the maximum value of the array, breaking ties randomly.
    """
    return np.random.choice(multi_argmax(arr))


def softmax_tau(
    arr: npt.NDArray[np.float64],
    tau: float,
) -> npt.NDArray[np.float64]:
    """
    Softmax function with temperature parameter `tau`.
    """
    res = arr / tau
    res = np.exp(res - np.max(res))  # np.max(res) is subtracted for stability
    res = res / np.sum(res)

    return res


def softmax_tau_batch(
    arr: tf.Tensor,  # shape: (n_samples, n_actions)
    tau: float,
) -> npt.NDArray[np.float64]:
    """
    Softmax function with temperature parameter `tau` over a batch of samples
    given as a TensorFlow tensor.
    """
    res = arr / tau

    # The row-wise maximum is subtracted for stability:
    res = tf.exp(res - tf.reduce_max(res, axis=1, keepdims=True))

    res = res / tf.reduce_sum(res, axis=1, keepdims=True)

    return res


class PolicyBase(ABC):
    @abstractmethod
    def get_action(self, state: StateType) -> int:
        pass

    @abstractmethod
    def get_probs(self, state: StateType) -> npt.NDArray[np.float64]:
        pass

    @abstractmethod
    def get_probs_from_Q_batch(
        self,
        Q_state_batch: tf.Tensor,  # shape: (n_samples, n_actions)
    ) -> npt.NDArray[np.float64]:  # Two-dimensional array
        """
        Gets the action selection probabilities from the action-values
        associated with a batch of states in a vectorized manner.

        Args:
            Q_state_batch:
                a two-dimensional array such that its (i, j) component is the
                estimated action-value of the j-th action when the agent is at
                the i-th item of a batch of states.

        Returns:
            A two-dimensional array such that its (i, j) component is the
            probability of selecting the j-th action when the agent is at
            the i-th item of a batch of states.
        """
        pass

    @abstractmethod
    def run_scheduler(self, step: int, episode: int) -> None:
        """
        Runs the scheduler(s) of the policy to vary its parameters during
        training.
        """
        pass

    def set_Q(self, Q: QApproximatorBase) -> None:
        """
        Sets the action-values approximator as an attribute of the object.
        """
        self.Q = Q


def check_Q_is_set(method):
    """
    Decorator that checks the `Q` attribute has been set before running a
    method.
    """
    @wraps(method)
    def decorated_method(self, *args, **kwargs):
        if self.Q is None:
            raise ValueError(
                'The attribute `Q` must be set before running the '
                f'`{method.__name__}` method.'
            )

        return method(self, *args, **kwargs)

    return decorated_method


class TabularActionsPolicyBase(PolicyBase, ABC):
    """
    Base class for policies associated with tabular-action environments; i.e.,
    environments where the actions are discrete and finite. The environment's
    states don't need to be finite or discrete as long as action-values for a
    given state can be obtained from the `Q` attribute using `Q[state]`.
    """
    def __init__(
        self,
        Q: Optional[QApproximatorBase] = None,
        *,
        env: Optional[gym.Env] = None,
        n_actions: Optional[int] = None,
    ) -> None:
        self.Q = Q

        if env is not None:
            self.n_actions = env.action_space.n
        elif n_actions is not None:
            self.n_actions = n_actions
        elif Q is not None:
            if hasattr(Q, 'shape'):
                self.n_actions = Q.shape[1]
            else:
                raise ShapelessQError()

    def set_Q(self, Q: QApproximatorBase) -> None:
        super().set_Q(Q)
        if hasattr(Q, 'shape'):
            self.n_actions = Q.shape[1]
        else:
            raise ShapelessQError()

    def _get_greedy_action(self, state: StateType) -> int:
        """
        This method is defined separately from the `get_greedy_action` method
        because the former shouldn't be decorated with `check_Q_is_set`, unlike
        the latter.
        """
        return random_argmax(self.Q[state])

    @check_Q_is_set
    def get_greedy_action(self, state: StateType) -> int:
        return self._get_greedy_action(state)

    @abstractmethod
    @check_Q_is_set
    def get_action(self, state: StateType) -> int:
        pass

    @abstractmethod
    def get_probs(self, state: StateType) -> npt.NDArray[np.float64]:
        """
        Returns an array with the probability of each action being selected by
        the policy at the given state.
        """
        pass

    @abstractmethod
    def get_probs_from_Q_batch(
        self,
        Q_state_batch: tf.Tensor,  # shape: (n_samples, n_actions)
    ) -> npt.NDArray[np.float64]:  # Two-dimensional array
        """
        Gets the action selection probabilities from the action-values
        associated with a batch of states in a vectorized manner.

        Args:
            Q_state_batch:
                a two-dimensional array such that its (i, j) component is the
                estimated action-value of the j-th action when the agent is at
                the i-th item of a batch of states.

        Returns:
            A two-dimensional array such that its (i, j) component is the
            probability of selecting the j-th action when the agent is at
            the i-th item of a batch of states.
        """
        pass

    @abstractmethod
    def run_scheduler(self, step: int, episode: int) -> None:
        """
        Runs the scheduler(s) of the policy to vary its parameters during
        training.
        """
        pass


class GreedyPolicy(TabularActionsPolicyBase):
    """
    A fully greedy policy.
    """
    def __init__(
        self,
        Q: Optional[QApproximatorBase] = None,
        *,
        env: Optional[gym.Env] = None,
        n_actions: Optional[int] = None
    ) -> None:
        super().__init__(Q, env=env, n_actions=n_actions)

    @check_Q_is_set
    def get_action(self, state: StateType) -> int:
        return self._get_greedy_action(state)

    @check_Q_is_set
    def get_probs(self, state: StateType) -> npt.NDArray[np.float64]:
        """
        Returns anb array with the probability of each action being selected by
        the policy at the given state.
        """
        max_indices = multi_argmax(self.Q[state])
        probs = np.zeros(self.n_actions)
        probs[max_indices] = 1. / max_indices.size

        return probs

    @check_Q_is_set
    def get_probs_from_Q_batch(
        self,
        Q_state_batch: tf.Tensor,  # shape: (n_samples, n_actions)
    ) -> npt.NDArray[np.float64]:  # Two-dimensional array
        """
        Gets the action selection probabilities from the action-values
        associated with a batch of states in a vectorized manner.

        Args:
            Q_state_batch:
                a two-dimensional array such that its (i, j) component is the
                estimated action-value of the j-th action when the agent is at
                the i-th item of a batch of states.

        Returns:
            A two-dimensional array such that its (i, j) component is the
            probability of selecting the j-th action when the agent is at
            the i-th item of a batch of states.
        """
        if not isinstance(self.Q, GradQApproximator):
            raise NotImplementedError(
                'the `get_probs_from_Q_batch` method can only be used if the '
                '`Q` attribute is a `GradQApproximator` object.'
            )

        probs = np.where(
            Q_state_batch == np.max(Q_state_batch, axis=1, keepdims=True),
            1.,
            0.
        )

        probs /= np.sum(probs, axis=1, keepdims=True)

        return probs

    def run_scheduler(self, step: int, episode: int) -> None:
        pass


class EpsGreedyPolicy(TabularActionsPolicyBase):
    """
    An epsilon-greedy policy.
    """
    def __init__(
        self,
        eps: Optional[float] = None,
        scheduler: Optional[Callable] = None,
        Q: Optional[QApproximatorBase] = None,
        *,
        env: Optional[gym.Env] = None,
        n_actions: Optional[int] = None
    ) -> None:
        super().__init__(Q, env=env, n_actions=n_actions)

        self.scheduler = scheduler

        if eps is None:
            if scheduler is None:
                raise ValueError(
                    'If the `eps` parameter is not set, then the scheduler '
                    'parameter must be.'
                )

            self.eps = scheduler(0, 0)
        else:
            self.eps = eps

    @check_Q_is_set
    def get_action(self, state: StateType) -> int:
        if np.random.random() < self.eps:
            return np.random.randint(self.n_actions)
        else:
            return self._get_greedy_action(state)

    @check_Q_is_set
    def get_probs(self, state: StateType) -> npt.NDArray[np.float64]:
        """
        Returns an array with the probability of each action being selected by
        the policy at the given state.
        """
        probs = np.full(
            shape=self.n_actions,
            fill_value=self.eps / self.n_actions,
            dtype=np.float64,
        )

        argmax_indices = multi_argmax(self.Q[state])

        for i in argmax_indices:
            probs[i] += (1 - self.eps) / argmax_indices.size

        return probs

    @check_Q_is_set
    def get_probs_from_Q_batch(
        self,
        Q_state_batch: tf.Tensor,  # shape: (n_samples, n_actions)
    ) -> npt.NDArray[np.float64]:  # Two-dimensional array
        """
        Gets the action selection probabilities from the action-values
        associated with a batch of states in a vectorized manner.

        Args:
            Q_state_batch:
                a two-dimensional array such that its (i, j) component is the
                estimated action-value of the j-th action when the agent is at
                the i-th item of a batch of states.

        Returns:
            A two-dimensional array such that its (i, j) component is the
            probability of selecting the j-th action when the agent is at
            the i-th item of a batch of states.
        """
        if not isinstance(self.Q, GradQApproximator):
            raise NotImplementedError(
                'the `get_probs_from_Q_batch` method can only be used if the '
                '`Q` attribute is a `GradQApproximator` object.'
            )
        mask = (Q_state_batch == np.max(Q_state_batch, axis=1, keepdims=True))

        probs = np.where(
            mask,
            1. - self.eps,
            0.
        )

        probs = (
            probs / np.sum(mask, axis=1, keepdims=True)
            + self.eps / probs.shape[1]
        )

        return probs

    def run_scheduler(self, step: int, episode: int) -> None:
        if self.scheduler:
            self.eps = self.scheduler(step, episode)


class SoftmaxPolicy(TabularActionsPolicyBase):
    """
    A policy that chooses actions with probability given by a softmax
    distribution with the action-values as preferences.
    """
    def __init__(
        self,
        tau: Optional[float] = None,  # temperature parameter
        scheduler: Optional[Callable] = None,
        Q: Optional[QApproximatorBase] = None,
        *,
        env: Optional[gym.Env] = None,
        n_actions: Optional[int] = None
    ) -> None:
        super().__init__(Q, env=env, n_actions=n_actions)

        self.scheduler = scheduler

        if tau is None:
            if scheduler is None:
                raise ValueError(
                    'If the `tau` parameter is not set, then the scheduler '
                    'parameter must be.'
                )

            self.tau = scheduler(0, 0)
        else:
            self.tau = tau

    @check_Q_is_set
    def get_action(self, state: StateType):
        return np.random.choice(
            range(self.n_actions),
            p=softmax_tau(self.Q[state], self.tau),
        )

    @check_Q_is_set
    def get_probs(self, state: StateType) -> npt.NDArray[np.float64]:
        """
        Returns an array with the probability of each action being selected by
        the policy at the given state.
        """
        probs = softmax_tau(self.Q[state], self.tau)
        return probs

    @check_Q_is_set
    def get_probs_from_Q_batch(
        self,
        Q_state_batch: tf.Tensor,  # shape: (n_samples, n_actions)
    ) -> npt.NDArray[np.float64]:  # Two-dimensional array
        """
        Gets the action selection probabilities from the action-values
        associated with a batch of states in a vectorized manner.

        Args:
            Q_state_batch:
                a two-dimensional array such that its (i, j) component is the
                estimated action-value of the j-th action when the agent is at
                the i-th item of a batch of states.

        Returns:
            A two-dimensional array such that its (i, j) component is the
            probability of selecting the j-th action when the agent is at
            the i-th item of a batch of states.
        """
        if not isinstance(self.Q, GradQApproximator):
            raise NotImplementedError(
                'the `get_probs_from_Q_batch` method can only be used if the '
                '`Q` attribute is a `GradQApproximator` object.'
            )

        probs = softmax_tau_batch(Q_state_batch, self.tau)
        return probs

    def run_scheduler(self, step: int, episode: int) -> None:
        if self.scheduler:
            self.tau = self.scheduler(step, episode)
