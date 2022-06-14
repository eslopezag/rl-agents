from abc import ABC, abstractmethod
from typing import Union, Tuple, Optional, Dict

import numpy as np
from numpy import typing as npt


class ExperienceBufferBase(ABC):
    def __init__(self, buffer_size: int, state_dim: int) -> None:
        self.buffer_size = buffer_size
        self.is_full = False
        self.last_index_added = -1
        self._buffer = {
            'prev_states': np.zeros(
                (buffer_size, state_dim), dtype=np.float64
            ),
            'prev_actions': np.zeros(buffer_size, dtype=np.int32),
            'rewards': np.zeros(buffer_size, dtype=np.float64),
            'next_states': np.zeros(
                (buffer_size, state_dim), dtype=np.float64
            ),
            'next_actions': np.zeros(buffer_size, dtype=np.int32),
            'next_is_terminal': np.full(buffer_size, False, dtype=bool),
        }

    @staticmethod
    def _transform_states(
        states: Tuple[Union[int, float, npt.NDArray[np.float64]]]
    ) -> Tuple[npt.NDArray[np.float64]]:
        """
        Transforms a tuple of states that might be integer or floats into the
        appropriate numpy array. This function assumes that all states passed
        are of the same data type.
        """
        if isinstance(states[0], (int, float)):
            return tuple(np.array(s, dtype=np.float64)[None] for s in states)
        elif isinstance(states[0], np.ndarray):
            return states

    def __getitem__(self, key) -> Dict[str, npt.NDArray[np.float64]]:
        if self.is_full:
            return self._buffer[key]
        else:
            return self._buffer[key][:self.last_index_added + 1]

    def _insert(
        self,
        index: int,
        prev_state: Union[int, float, npt.NDArray[np.float64]],
        prev_action: int,
        reward: float,
        next_state: Union[int, float, npt.NDArray[np.float64]],
        next_action: int,
        next_is_terminal: bool,
    ) -> None:
        self._buffer['prev_states'][index] = prev_state
        self._buffer['prev_actions'][index] = prev_action
        self._buffer['rewards'][index] = reward
        self._buffer['next_states'][index] = next_state
        self._buffer['next_actions'][index] = next_action
        self._buffer['next_is_terminal'][index] = next_is_terminal

    @property
    def used_buffer_size(self):
        """
        Returns the number of experience items currently stored in the buffer,
        which can be less than its total capacity, given by `self.buffer_size`.
        """
        if self.is_full:
            return self.buffer_size
        else:
            return self.last_index_added + 1

    def get_sample(
        self, size: Optional[int] = None
    ) -> Dict[str, npt.NDArray[np.float64]]:
        """
        Gets a random sample of the experience buffer with the given size. If
        `size` is `None`, the whole buffer is returned.
        """
        used_buffer_size = self.used_buffer_size

        if size is None or size > used_buffer_size:
            return {key: self[key] for key in self._buffer}

        # Get random indices:
        indices = np.random.choice(used_buffer_size, size, replace=False)
        return {key: self[key][indices] for key in self._buffer}

    @abstractmethod
    def add(
        self,
        prev_state: Union[int, float, npt.NDArray[np.float64]],
        prev_action: int,
        reward: float,
        next_state: Union[int, float, npt.NDArray[np.float64]],
        next_action: int,
        next_is_terminal: bool,
    ) -> None:
        pass


class OrderedExperienceBuffer(ExperienceBufferBase):
    def __init__(self, buffer_size: int, state_dim: int) -> None:
        super().__init__(buffer_size, state_dim)

    def add(
        self,
        prev_state: Union[int, float, npt.NDArray[np.float64]],
        prev_action: int,
        reward: float,
        next_state: Union[int, float, npt.NDArray[np.float64]],
        next_action: int,
        next_is_terminal: bool,
    ) -> None:
        prev_state, next_state = self._transform_states(
            (prev_state, next_state)
        )

        self.last_index_added = (
            (self.last_index_added + 1) % self.buffer_size
        )

        if not self.is_full and self.last_index_added == self.buffer_size - 1:
            self.is_full = True

        self._insert(
            self.last_index_added,
            prev_state,
            prev_action,
            reward,
            next_state,
            next_action,
            next_is_terminal,
        )


class RandomExperienceBuffer(ExperienceBufferBase):
    def __init__(self, buffer_size: int, state_dim: int) -> None:
        super().__init__(buffer_size, state_dim)

    def add(
        self,
        prev_state: Union[int, float, npt.NDArray[np.float64]],
        prev_action: int,
        reward: float,
        next_state: Union[int, float, npt.NDArray[np.float64]],
        next_action: int,
        next_is_terminal: bool,
    ) -> None:
        prev_state, next_state = self._transform_states(
            (prev_state, next_state)
        )

        self.last_index_added = (
            (self.last_index_added + 1) % self.buffer_size
        )

        if not self.is_full and self.last_index_added == self.buffer_size - 1:
            self.is_full = True

        if self.is_full:
            index = np.random.randint(self.buffer_size)
        else:
            index = self.last_index_added

        self._insert(
            index,
            prev_state,
            prev_action,
            reward,
            next_state,
            next_action,
            next_is_terminal,
        )
