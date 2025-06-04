import collections
import copy
import datetime
import gym
import gymnasium
import numpy as np
import os
import torch

from typing import Any, Mapping, Optional, Tuple, Union

from isaaclab_rl.algorithms import config, logger
from isaaclab_rl.algorithms.memories import Memory


class Agent:
    def __init__(
        self,
        memory: Optional[Union[Memory, Tuple[Memory]]] = None,
        observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
        action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        """Base class that represent a RL agent

        :param models: Models used by the agent
        :type models: dictionary of skrl.models.torch.Model
        :param memory: Memory to storage the transitions.
                       If it is a tuple, the first element will be used for training and
                       for the rest only the environment transitions will be added
        :type memory: skrl.memory.torch.Memory, list of skrl.memory.torch.Memory or None
        :param observation_space: Observation/state space or shape (default: ``None``)
        :type observation_space: int, tuple or list of int, gym.Space, gymnasium.Space or None, optional
        :param action_space: Action space or shape (default: ``None``)
        :type action_space: int, tuple or list of int, gym.Space, gymnasium.Space or None, optional
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or torch.device, optional
        :param cfg: Configuration dictionary
        :type cfg: dict
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.cfg = cfg if cfg is not None else {}
        self.device = (
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
        )

        if type(memory) is list:
            self.memory = memory[0]
            self.secondary_memories = memory[1:]
        else:
            self.memory = memory
            self.secondary_memories = []

        # self.tracking_data = collections.defaultdict(list)
        self.tb_log = self.cfg.get("experiment", {}).get("tb_log", False)
        if self.tb_log:
            print("Saving stuff to tensorboard")

        self._random_distribution = torch.distributions.uniform.Uniform(
                low=torch.tensor(-1, device=self.device, dtype=torch.float32),
                high=torch.tensor(1, device=self.device, dtype=torch.float32),
        )


        self.training = True

        


    def __str__(self) -> str:
        """Generate a representation of the agent as string

        :return: Representation of the agent as string
        :rtype: str
        """
        string = f"Agent: {repr(self)}"
        for k, v in self.cfg.items():
            if type(v) is dict:
                string += f"\n  |-- {k}"
                for k1, v1 in v.items():
                    string += f"\n  |     |-- {k1}: {v1}"
            else:
                string += f"\n  |-- {k}: {v}"
        return string

    def _empty_preprocessor(self, _input: Any, *args, **kwargs) -> Any:
        """Empty preprocess method

        This method is defined because PyTorch multiprocessing can't pickle lambdas

        :param _input: Input to preprocess
        :type _input: Any

        :return: Preprocessed input
        :rtype: Any
        """
        return _input

    

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        """Initialize the agent

        This method should be called before the agent is used.
        It will initialize the TensorBoard writer (and optionally Weights & Biases) and create the checkpoints directory

        :param trainer_cfg: Trainer configuration
        :type trainer_cfg: dict, optional
        """
        trainer_cfg = trainer_cfg if trainer_cfg is not None else {}

    def random_act(
        self, inputs) -> Tuple[torch.Tensor, None, Mapping[str, Union[torch.Tensor, Any]]]:
        """Act randomly according to the action space

        :param inputs: Model inputs. The most common keys are:

                       - ``"states"``: state of the environment used to make the decision
                       - ``"taken_actions"``: actions taken by the policy for the given states
        :type inputs: dict where the values are typically torch.Tensor
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        :raises NotImplementedError: Unsupported action space

        :return: Model output. The first component is the action to be taken by the agent
        :rtype: tuple of torch.Tensor, None, and dict
        """           

        return (
            self._random_distribution.sample(sample_shape=(inputs.shape[0], self.num_actions)),
            None,
            {},
        )

    def save(self, path: str) -> None:
        """Save the agent to the specified path

        :param path: Path to save the model to
        :type path: str
        """
        modules = {}
        for name, module in self.checkpoint_modules.items():
            modules[name] = self._get_internal_value(module)
        torch.save(modules, path)

    def load(self, path: str) -> None:
        """Load the model from the specified path

        The final storage device is determined by the constructor of the model

        :param path: Path to load the model from
        :type path: str
        """
        modules = torch.load(path, map_location=self.device)
        if type(modules) is dict:
            for name, data in modules.items():
                module = self.checkpoint_modules.get(name, None)
                if module is not None:
                    if hasattr(module, "load_state_dict"):
                        module.load_state_dict(data)
                        if hasattr(module, "eval"):
                            module.eval()
                    else:
                        raise NotImplementedError
                else:
                    logger.warning(f"Cannot load the {name} module. The agent doesn't have such an instance")

