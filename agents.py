from typing import Optional, Union, Tuple
from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
from numpy import typing as npt
import gym
from tqdm import tqdm
import dill

from agent_history import AgentHistoryBase, TabularAgentHistory
from q_approximators import QApproximatorBase, TabularQApproximator
from policies import PolicyBase, GreedyPolicy


class AgentBase(ABC):
    def __init__(
        self,
        env: gym.Env,
        Q: QApproximatorBase,
        history: AgentHistoryBase,
        step_on_episode_start: bool,
        target_policy: PolicyBase,
        exploration_policy: PolicyBase,
        discount: float = 1.,
        mode: str = 'training',
        output_filename: Optional[str] = None,
    ) -> None:

        self.env = env
        self.Q = Q
        self.history = history

        # Set the flag that specifies whether the firsty transition of an
        # episode counts as a step:
        self.step_on_episode_start = step_on_episode_start

        self.target_policy = target_policy
        self.target_policy.set_Q(Q)

        self.exploration_policy = exploration_policy
        self.exploration_policy.set_Q(Q)

        self.discount = discount

        if output_filename:
            self.output_filename = output_filename
        else:
            self.output_filename = '{agent}_{timestamp}'.format(
                agent=self.__class__.__qualname__,
                timestamp=datetime.now().timestamp()
            )

        if mode == 'training' or mode == 'inference':
            self.mode = mode
        else:
            raise ValueError(
                'The agent\'s mode can only be "training" or "inference."'
            )

        self.last_state = None
        self.last_action = None
        self.last_done = False  # Flags whether last step ended episode

        self.step = -1
        self.episode = -1

    def set_mode(self, new_mode: str) -> None:
        if new_mode == 'training' or new_mode == 'inference':
            self.mode = new_mode
        else:
            raise ValueError(
                'The agent\'s mode can only be "training" or "inference."'
            )

    def get_action(
        self,
        state: Union[int, float, npt.NDArray[np.float64]],
    ) -> Union[int, float, npt.NDArray[np.float64]]:
        if self.mode == 'inference':
            return self.target_policy.get_greedy_action(state)
        elif self.mode == 'training':
            return self.exploration_policy.get_action(state)

    def _register_history(self, reward: Union[float, int], done: bool) -> None:
        self.history.register_reward(reward)
        self.history.register_Q(self.Q)

        if done:
            self.history.register_episode_end()

    @abstractmethod
    def train_step(
        self,
        state: Union[int, float, npt.NDArray[np.float64]],
        reward: Union[float, int],
    ) -> Union[int, float, npt.NDArray[np.float64]]:
        pass

    def _reset_env(self) -> Union[int, float, npt.NDArray[np.float64]]:
        observation = self.env.reset()
        return observation

    def _set_episode_start(self) -> Tuple[
        Union[int, float, npt.NDArray[np.float64]],
        Union[float, int],
        bool,
    ]:
        self.episode += 1
        self.last_state = None
        self.last_action = None
        self.last_done = False
        observation = self._reset_env()
        reward = None
        done = False

        return observation, reward, done

    def _step_update(self, progress_bar: tqdm):
        """
        Determines the logic used to increase the step count.
        """
        if self.step_on_episode_start or (self.last_state is not None):
            self.step += 1
            progress_bar.update()

    @abstractmethod
    def run_policies_schedulers(self) -> None:
        pass

    def train(self, training_steps: int) -> None:
        with tqdm(total=training_steps) as progress_bar:
            while progress_bar.n < training_steps:
                observation, reward, done = self._set_episode_start()
                self._register_history(reward, done)

                while not self.last_done:
                    self._step_update(progress_bar)
                    self.Q.run_scheduler(self.step, self.episode)
                    self.run_policies_schedulers()
                    action = self.train_step(observation, reward)

                    if done:
                        self.last_done = True
                    else:
                        observation, reward, done, info = self.env.step(action)
                        self._register_history(reward, done)

                    if progress_bar.n >= training_steps:
                        break

        self.env.close()

        print(self.Q)

        # This step seems to be necessary so that dill properly saves the `Q`
        # attribute of the object in the call to `dill.dump`:
        dill.dumps(self.Q, byref=False, recurse=True)

        with open(f'{self.output_filename}.dill', 'wb') as fopen:
            dill.dump(self, fopen, byref=False, recurse=True)

    def show_training_results(self) -> None:
        self.history.show_training_results()


class TabularSarsaAgent(AgentBase):
    def __init__(
        self,
        env: gym.Env,
        Q: TabularQApproximator,
        history: TabularAgentHistory,
        target_policy: PolicyBase,
        discount: float = 1,
        mode: str = 'training',
        output_filename: Optional[str] = None,
    ) -> None:
        super().__init__(
            env=env,
            Q=Q,
            history=history,
            step_on_episode_start=False,
            target_policy=target_policy,
            exploration_policy=target_policy,
            discount=discount,
            mode=mode,
            output_filename=output_filename,
        )
        self.policy = target_policy

    def get_action(self, state) -> int:
        return super().get_action(state)

    def _reset_env(self) -> Union[int, float, npt.NDArray[np.float64]]:
        return super()._reset_env()

    def run_policies_schedulers(self) -> None:
        self.policy.run_scheduler(self.step, self.episode)

    def train_step(
        self,
        state: Union[int, float, npt.NDArray[np.float64]],
        reward: Union[float, int],
    ) -> Union[int, float, npt.NDArray[np.float64]]:
        if self.mode != 'training':
            raise Exception('The agent cannot be trained in inference mode.')

        # Get action from policy:
        action = self.get_action(state)

        if self.last_state is not None and self.last_action is not None:
            target = reward + self.discount * self.Q[state, action]
            self.Q.update(self.last_state, self.last_action, target)

        self.last_state = state
        self.last_action = action

        return action


class TabularQLearningAgent(AgentBase):
    def __init__(
        self,
        env: gym.Env,
        Q: TabularQApproximator,
        history: TabularAgentHistory,
        exploration_policy: PolicyBase,
        discount: float = 1,
        mode: str = 'training',
        output_filename: Optional[str] = None,
    ) -> None:
        super().__init__(
            env=env,
            Q=Q,
            history=history,
            step_on_episode_start=False,
            target_policy=GreedyPolicy(),
            exploration_policy=exploration_policy,
            discount=discount,
            mode=mode,
            output_filename=output_filename,
        )

    def get_action(self, state) -> int:
        return super().get_action(state)

    def _reset_env(self) -> Union[int, float, npt.NDArray[np.float64]]:
        return super()._reset_env()

    def run_policies_schedulers(self) -> None:
        self.exploration_policy.run_scheduler(self.step, self.episode)

    def train_step(
        self,
        state: Union[int, float, npt.NDArray[np.float64]],
        reward: Union[float, int],
    ) -> Union[int, float, npt.NDArray[np.float64]]:
        if self.mode != 'training':
            raise Exception('The agent cannot be trained in inference mode.')

        # Get action from policy:
        action = self.get_action(state)

        if self.last_state is not None and self.last_action is not None:
            target = reward + self.discount * np.max(self.Q[state])
            self.Q.update(self.last_state, self.last_action, target)

        self.last_state = state
        self.last_action = action

        return action


class TabularExpectedSarsaAgent(AgentBase):
    def __init__(
        self,
        env: gym.Env,
        Q: TabularQApproximator,
        history: TabularAgentHistory,
        target_policy: PolicyBase,
        exploration_policy: PolicyBase,
        discount: float = 1.,
        mode: str = 'training',
        output_filename: Optional[str] = None,
    ) -> None:
        super().__init__(
            env=env,
            Q=Q,
            history=history,
            step_on_episode_start=False,
            target_policy=target_policy,
            exploration_policy=exploration_policy,
            discount=discount,
            mode=mode,
            output_filename=output_filename,
        )

    def get_action(self, state) -> int:
        return super().get_action(state)

    def _reset_env(self) -> Union[int, float, npt.NDArray[np.float64]]:
        return super()._reset_env()

    def run_policies_schedulers(self) -> None:
        self.target_policy.run_scheduler(self.step, self.episode)
        self.exploration_policy.run_scheduler(self.step, self.episode)

    def train_step(
        self,
        state: Union[int, float, npt.NDArray[np.float64]],
        reward: Union[float, int],
    ) -> Union[int, float, npt.NDArray[np.float64]]:
        if self.mode != 'training':
            raise Exception('The agent cannot be trained in inference mode.')

        # Get action from policy:
        action = self.get_action(state)

        if self.last_state is not None and self.last_action is not None:
            target = (
                reward + self.discount *
                np.dot(self.target_policy.get_probs(state), self.Q[state])
            )
            self.Q.update(self.last_state, self.last_action, target)

        self.last_state = state
        self.last_action = action

        return action
