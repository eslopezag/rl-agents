from abc import ABC, abstractmethod
from typing import Union
from itertools import chain

import numpy as np
from matplotlib import pyplot as plt

from q_approximators import QApproximatorBase, TabularQApproximator


class AgentHistoryBase(ABC):
    @abstractmethod
    def register_reward(self, reward: Union[int, float]) -> None:
        pass

    @abstractmethod
    def register_Q(self, Q: QApproximatorBase) -> None:
        pass

    @abstractmethod
    def register_episode_end(self) -> None:
        pass

    @abstractmethod
    def show_training_results(self) -> None:
        pass


class TabularAgentHistory(AgentHistoryBase):
    def __init__(self) -> None:
        # Initialize the list that will contain the indices of the time steps
        # where episodes ended:
        self.episode_ends = []

        # Initialize the list that will keep a history of the rewards:
        self.reward_history = []

    def register_reward(self, reward: Union[int, float]) -> None:
        self.reward_history.append(reward)

    def register_Q(self, Q: TabularQApproximator) -> None:
        pass

    def register_episode_end(self) -> None:
        self.episode_ends.append(len(self.reward_history) - 1)

    def show_training_results(self) -> None:
        fig, ax = plt.subplots(figsize=(10, 21))

        ax.set_xlabel('Episode')
        ax.set_ylabel(
            'Running average of episode reward (window size = 100)'
        )
        episode_rewards = [
            sum(r for r in self.reward_history[i + 1: j + 1] if r is not None)
            for i, j in zip(
                chain([-1], self.episode_ends),
                self.episode_ends,
            )
        ]
        cumsum = np.cumsum(np.insert(episode_rewards, 0, 0))
        running_avg = (cumsum[100:] - cumsum[:-100]) / 100
        ax.plot(range(100, len(running_avg) + 100), running_avg)

        plt.show(block=True)
