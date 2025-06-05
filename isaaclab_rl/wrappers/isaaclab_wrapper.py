import gymnasium
import torch
from collections.abc import Sequence
from typing import Any, Mapping, Tuple, Union

from skrl import config


class IsaacLabWrapper(object):
    def __init__(self, env: Any) -> None:
        """Isaac Lab environment wrapper

        :param env: The environment to wrap
        :type env: Any supported Isaac Lab environment
        """
        self._env = env
        try:
            self._unwrapped = self._env.unwrapped
        except:
            self._unwrapped = env

        # device
        if hasattr(self._unwrapped, "device"):
            self._device = config.torch.parse_device(self._unwrapped.device)
        else:
            self._device = config.torch.parse_device(None)

        self._reset_once = True
        self._observations = None
        self._info = {}
        self.obs_stack = self._unwrapped.obs_stack

    def __getattr__(self, key: str) -> Any:
        """Get an attribute from the wrapped environment

        :param key: The attribute name
        :type key: str

        :raises AttributeError: If the attribute does not exist

        :return: The attribute value
        :rtype: Any
        """
        if hasattr(self._env, key):
            return getattr(self._env, key)
        if hasattr(self._unwrapped, key):
            return getattr(self._unwrapped, key)
        raise AttributeError(
            f"Wrapped environment ({self._unwrapped.__class__.__name__}) does not have attribute '{key}'"
        )

    def get_observations(self):
        try:
            self._env.get_observations()
        except:
            self._unwrapped.get_observations()
        return

    def configure_gym_env_spaces(self, obs_stack=1):
        return self._env.configure_gym_env_spaces(obs_stack)

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

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: torch.Tensor

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of torch.Tensor and any other info
        """
        self._observations, reward, terminated, truncated, self._info = self._env.step(actions)
        return self._observations, reward.view(-1, 1), terminated.view(-1, 1), truncated.view(-1, 1), self._info

    def reset(self, env_ids=None, hard: bool = False) -> Tuple[torch.Tensor, Any]:
        """Reset the environment

        :return: Observation, info
        :rtype: torch.Tensor and any other info
        """
        # self._env._reset_idx() env_ids: Sequence[int],
        if env_ids is not None:
            # this is a mix of code compied from DirectRLEnv
            self._unwrapped.scene.reset(env_ids)
            self._unwrapped.episode_length_buf[env_ids] = 0

            # update observations
            obs = self._unwrapped.get_observations()

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

    @property
    def device(self) -> torch.device:
        """The device used by the environment

        If the wrapped environment does not have the ``device`` property, the value of this property
        will be ``"cuda"`` or ``"cpu"`` depending on the device availability
        """
        return self._device

    @property
    def num_envs(self) -> int:
        """Number of environments

        If the wrapped environment does not have the ``num_envs`` property, it will be set to 1
        """
        return self._unwrapped.num_envs if hasattr(self._unwrapped, "num_envs") else 1

    @property
    def num_agents(self) -> int:
        """Number of agents

        If the wrapped environment does not have the ``num_agents`` property, it will be set to 1
        """
        return self._unwrapped.num_agents if hasattr(self._unwrapped, "num_agents") else 1

    @property
    def state_space(self) -> Union[gymnasium.Space, None]:
        """State space

        If the wrapped environment does not have the ``state_space`` property, ``None`` will be returned
        """
        return self._unwrapped.state_space if hasattr(self._unwrapped, "state_space") else None

    @property
    def observation_space(self) -> gymnasium.Space:
        """Observation space"""
        return self._unwrapped.single_observation_space

    @property
    def action_space(self) -> gymnasium.Space:
        """Action space"""
        return self._unwrapped.single_action_space
