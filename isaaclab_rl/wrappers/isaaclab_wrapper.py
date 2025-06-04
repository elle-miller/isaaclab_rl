import gymnasium
import torch
from typing import Any, Mapping, Tuple, Union

# from skrl.envs.wrappers.torch.base import Wrapper
from isaaclab_rl.wrappers.base_wrapper import Wrapper

from collections.abc import Sequence

class IsaacLabWrapper(Wrapper):
    def __init__(self, env: Any) -> None:
        """Isaac Lab environment wrapper

        :param env: The environment to wrap
        :type env: Any supported Isaac Lab environment
        """
        super().__init__(env)

        self._reset_once = True
        self._observations = None
        self._info = {}
        self.obs_stack = self._env.unwrapped.obs_stack

    @property
    def state_space(self) -> Union[gymnasium.Space, None]:
        """State space"""
        try:
            return self._unwrapped.single_observation_space["critic"]
        except KeyError:
            pass
        try:
            return self._unwrapped.state_space
        except AttributeError:
            return None

    # TODO: REDEFINING THESE CAUSES ISSUES!

    # @property
    # def observation_space(self) -> gymnasium.Space:
    #     """Observation space
    #     """
    #     try:
    #         return self._unwrapped.single_observation_space
    #     except:
    #         return self._unwrapped.observation_space

    # @property
    # def action_space(self) -> gymnasium.Space:
    #     """Action space
    #     """
    #     try:
    #         return self._unwrapped.single_action_space
    #     except:
    #         return self._unwrapped.action_space

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: torch.Tensor

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of torch.Tensor and any other info
        """
        self._observations, reward, terminated, truncated, self._info = self._env.step(actions)
        return self._observations, reward.view(-1, 1), terminated.view(-1, 1), truncated.view(-1, 1), self._info

    def reset(self, env_ids = None, hard: bool = False) -> Tuple[torch.Tensor, Any]:
        """Reset the environment

        :return: Observation, info
        :rtype: torch.Tensor and any other info
        """
        # self._env._reset_idx() env_ids: Sequence[int], 
        if env_ids is not None:
            # this is a mix of code compied from DirectRLEnv
            self._env.unwrapped.scene.reset(env_ids)
            self._env.unwrapped.episode_length_buf[env_ids] = 0
            
            # update observations
            obs = self._env.unwrapped.get_observations()

            # if we are frame stacking need to do something special
            if self.obs_stack != 1:
                obs = self.get_reset_obs(obs)

            return obs, self._info

        if hard:
            self._observations, self._info = self._env.reset()
        else:
            if self._reset_once:
                self._observations, self._info = self._env.reset()
                self._reset_once = False
        return self._observations, self._info

    def render(self, *args, **kwargs) -> None:
        """Render the environment"""
        return None

    def close(self) -> None:
        """Close the environment"""
        self._env.close()
