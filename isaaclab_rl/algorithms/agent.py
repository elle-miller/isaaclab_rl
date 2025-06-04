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
from isaaclab_rl.algorithms.models import Model


class Agent:
    def __init__(
        self,
        models: Mapping[str, Model],
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
        self.models = models
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

        # convert the models to their respective device
        for model in self.models.values():
            if model is not None:
                model.to(model.device)

        # self.tracking_data = collections.defaultdict(list)
        self.tb_log = self.cfg.get("experiment", {}).get("tb_log", False)
        if self.tb_log:
            print("Saving stuff to tensorboard")

        self._random_distribution = torch.distributions.uniform.Uniform(
                low=torch.tensor(-1, device=self.device, dtype=torch.float32),
                high=torch.tensor(1, device=self.device, dtype=torch.float32),
        )

        # self._track_rewards = collections.deque(maxlen=100)
        # self._track_timesteps = collections.deque(maxlen=100)
        # self._cumulative_rewards = None
        # self._cumulative_timesteps = None

        self.training = True

        # checkpoint
        self.checkpoint_modules = {}
        self.checkpoint_store_separately = self.cfg.get("experiment", {}).get("store_separately", False)
        self.checkpoint_best_modules = {"timestep": 0, "reward": -(2**31), "saved": False, "modules": {}}

        # experiment directory
        directory = self.cfg.get("experiment", {}).get("directory", "")
        experiment_name = self.cfg.get("experiment", {}).get("experiment_name", "")
        if not directory:
            directory = os.path.join(os.getcwd(), "runs")
        if not experiment_name:
            raise ValueError

        self.experiment_dir = os.path.join(directory, experiment_name)

        os.makedirs(os.path.join(self.experiment_dir, "checkpoints"), exist_ok=True)


        # set up distributed runs
        if config.torch.is_distributed:
            logger.info(
                f"Distributed (rank: {config.torch.rank}, local rank: {config.torch.local_rank}, world size: {config.torch.world_size})"
            )
            torch.distributed.init_process_group("nccl", rank=config.torch.rank, world_size=config.torch.world_size)
            torch.cuda.set_device(config.torch.local_rank)

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

    def _get_internal_value(self, _module: Any) -> Any:
        """Get internal module/variable state/value

        :param _module: Module or variable
        :type _module: Any

        :return: Module/variable state/value
        :rtype: Any
        """
        return _module.state_dict() if hasattr(_module, "state_dict") else _module

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


    def write_checkpoint(self, mean_eval_return: float, timestep: int, timesteps: int) -> None:
        """Write checkpoint (modules) to disk

        The checkpoints are saved in the directory 'checkpoints' in the experiment directory.
        The name of the checkpoint is the current timestep if timestep is not None, otherwise it is the current time.

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        # checkpoint_modules is a dict of "policy", "value", and "optimiser"
        if mean_eval_return > self.checkpoint_best_modules["reward"]:
            self.checkpoint_best_modules["timestep"] = timestep
            self.checkpoint_best_modules["reward"] = mean_eval_return
            self.checkpoint_best_modules["saved"] = False
            self.checkpoint_best_modules["modules"] = {
                k: copy.deepcopy(self._get_internal_value(v)) for k, v in self.checkpoint_modules.items()
            }

        tag = str(timestep if timestep is not None else datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f"))
        # separated modules
        if self.checkpoint_store_separately:
            for name, module in self.checkpoint_modules.items():
                torch.save(
                    self._get_internal_value(module),
                    os.path.join(self.experiment_dir, "checkpoints", f"{name}_{tag}.pt"),
                )
        # whole agent
        else:
            modules = {}
            for name, module in self.checkpoint_modules.items():
                modules[name] = self._get_internal_value(module)
            print("saving", f"agent_{tag}.pt")
            torch.save(modules, os.path.join(self.experiment_dir, "checkpoints", f"agent_{tag}.pt"))

        # best modules
        if self.checkpoint_best_modules["modules"] and not self.checkpoint_best_modules["saved"]:
            # separated modules
            if self.checkpoint_store_separately:
                for name, module in self.checkpoint_modules.items():
                    torch.save(
                        self.checkpoint_best_modules["modules"][name],
                        os.path.join(self.experiment_dir, "checkpoints", f"best_{name}.pt"),
                    )
                    print(
                        "Separate: New best reward, saving to",
                        os.path.join(self.experiment_dir, "checkpoints", "best_agent.pt"),
                    )
            # whole agent
            else:
                modules = {}
                for name, module in self.checkpoint_modules.items():
                    modules[name] = self.checkpoint_best_modules["modules"][name]
                torch.save(modules, os.path.join(self.experiment_dir, "checkpoints", f"best_agent.pt"))
                print("New best reward, saving to best_agent.pt")

            self.checkpoint_best_modules["saved"] = True



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

