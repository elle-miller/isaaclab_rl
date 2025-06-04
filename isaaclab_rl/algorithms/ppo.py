import copy
import gym
import gymnasium
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Mapping, Optional, Tuple, Union
from torch import nn
import kornia
import numpy as np
from isaaclab_rl.algorithms.agent import Agent
from isaaclab_rl.algorithms.memories import Memory
from isaaclab_rl.algorithms.models import Model
import gc
import optuna

from torch.amp import GradScaler, autocast


# [start-config-dict-torch]
PPO_DEFAULT_CONFIG = {
    "rollouts": 16,  # number of rollouts before updating
    "learning_epochs": 8,  # number of learning epochs during each update
    "mini_batches": 2,  # number of mini batches during each learning epoch
    "discount_factor": 0.99,  # discount factor (gamma)
    "lambda": 0.95,  # TD(lambda) coefficient (lam) for computing returns and advantages
    "learning_rate": 1e-3,  # learning rate
    "learning_rate_scheduler": None,  # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},  # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})
    "state_preprocessor": None,  # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},  # state preprocessor's kwargs (e.g. {"size": env.observation_space})
    "value_preprocessor": None,  # value preprocessor class (see skrl.resources.preprocessors)
    "value_preprocessor_kwargs": {},  # value preprocessor's kwargs (e.g. {"size": 1})
    "random_timesteps": 0,  # random exploration steps
    "learning_starts": 0,  # learning starts after this many steps
    "grad_norm_clip": 0.5,  # clipping coefficient for the norm of the gradients
    "ratio_clip": 0.2,  # clipping coefficient for computing the clipped surrogate objective
    "value_clip": 0.2,  # clipping coefficient for computing the value loss (if clip_predicted_values is True)
    "clip_predicted_values": False,  # clip predicted values during value loss computation
    "entropy_loss_scale": 0.0,  # entropy loss scaling factor
    "value_loss_scale": 1.0,  # value loss scaling factor
    "kl_threshold": 0,  # KL divergence threshold for early stopping
    "rewards_shaper": None,  # rewards shaping function: Callable(reward, timestep, timesteps) -> reward
    "time_limit_bootstrap": False,  # bootstrap at timeout termination (episode truncation)
    "experiment": {
        "directory": "",  # experiment's parent directory
        "experiment_name": "",  # experiment name
        "store_separately": False,  # whether to store checkpoints separately
        "wandb": False,  # whether to use Weights & Biases
        "wandb_kwargs": {},  # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    },
}
# [end-config-dict-torch]


class PPO(Agent):
    def __init__(
        self,
        encoder,
        value_preprocessor,
        models: Mapping[str, Model],
        memory: Optional[Union[Memory, Tuple[Memory]]] = None,
        observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
        action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[dict] = None,
        auxiliary_task=None,
        writer=None,

    ) -> None:
        """Proximal Policy Optimization (PPO)

        https://arxiv.org/abs/1707.06347

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

        :raises KeyError: If the models dictionary is missing a required key
        """
        _cfg = copy.deepcopy(PPO_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            cfg=_cfg,
        )

        self.writer = writer
        self.wandb_session = writer.wandb_session
        self.tb_writer = writer.tb_writer

        # hyperparams
        self._learning_epochs = self.cfg["learning_epochs"]
        self._mini_batches = self.cfg["mini_batches"]
        self._rollouts = self.cfg["rollouts"]
        self._rollout = 0
        self._grad_norm_clip = self.cfg["grad_norm_clip"]
        self._ratio_clip = self.cfg["ratio_clip"]
        self._value_clip = self.cfg["value_clip"]
        self._clip_predicted_values = self.cfg["clip_predicted_values"]
        self._value_loss_scale = self.cfg["value_loss_scale"]
        self._entropy_loss_scale = self.cfg["entropy_loss_scale"]
        self._kl_threshold = self.cfg["kl_threshold"]
        self._learning_rate = self.cfg["learning_rate"]
        self._learning_rate_scheduler = self.cfg["learning_rate_scheduler"]
        self._value_preprocessor = value_preprocessor
        self._discount_factor = self.cfg["discount_factor"]
        self._lambda = self.cfg["lambda"]
        self._rewards_shaper = self.cfg["rewards_shaper"]
        self._time_limit_bootstrap = self.cfg["time_limit_bootstrap"]

        # models
        self.policy = self.models.get("policy", None)
        self.value = self.models.get("value", None)
        self.encoder = encoder

        self.policy.eval()
        self.value.eval()
        self.encoder.eval()

        self.scaler = GradScaler()

        self.auxiliary_task = auxiliary_task

        # Create separate optimizers for different network components
        self.policy_optimiser = torch.optim.Adam(self.policy.parameters(), lr=self._learning_rate)
        self.value_optimiser = torch.optim.Adam(self.value.parameters(), lr=self._learning_rate)
        self.encoder_optimiser = torch.optim.Adam(self.encoder.parameters(), lr=self._learning_rate)

        # checkpoint models
        if self.writer.save_checkpoints > 0:
            self.writer.checkpoint_modules["policy"] = self.policy
            self.writer.checkpoint_modules["value"] = self.value
            self.writer.checkpoint_modules["encoder"] = self.encoder
        
            # set up preprocessors
            if self.encoder.state_preprocessor is not None:
                self.writer.checkpoint_modules["state_preprocessor"] = self.encoder.state_preprocessor

            self.writer.checkpoint_modules["policy_optimiser"] = self.policy_optimiser
            self.writer.checkpoint_modules["value_optimiser"] = self.value_optimiser
            self.writer.checkpoint_modules["encoder_optimiser"] = self.encoder_optimiser

            if self._value_preprocessor is not None:
                self.writer.checkpoint_modules["value_preprocessor"] = self._value_preprocessor

        
        self.num_actions = self.action_space.shape[0]

        if self._learning_rate_scheduler is not None:
            self.scheduler = self._learning_rate_scheduler(
                self.optimiser, **self.cfg["learning_rate_scheduler_kwargs"]
            )

        self.update_step = 0
        self.epoch_step = 0

        # set up automatic mixed precision
        self._mixed_precision = True
        self._device_type = torch.device(device).type
        self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self._mixed_precision)
        

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        """Initialize the agent. The  trainer calls this.
        Why is there a methods called "init">?>>?><>?
        """
        super().init(trainer_cfg=trainer_cfg)

        dtype = torch.float32

        # create tensors in memory
        if self.memory is not None:
            print("****RL Agent Memory****")
            self.observation_names = []
            for k, v in self.observation_space["policy"].items():
                print(f"PPO: {k}: tensor size {v.shape}")
                self.memory.create_tensor(
                    name=k,
                    size=v.shape,
                    dtype=torch.uint8 if k == "pixels" else dtype,
                )
                self.observation_names.append(k)

            # self.memory.create_tensor(name="states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=dtype)
            self.memory.create_tensor(name="rewards", size=1, dtype=dtype)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="truncated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="log_prob", size=1, dtype=dtype)
            self.memory.create_tensor(name="values", size=1, dtype=dtype)
            self.memory.create_tensor(name="returns", size=1, dtype=dtype)
            self.memory.create_tensor(name="advantages", size=1, dtype=dtype)

            self._tensors_names = self.observation_names + [
                    "actions",
                    "log_prob",
                    "values",
                    "returns",
                    "advantages",
                ]

        # create temporary variables needed for storage and computation
        self._current_next_states = None

    def record_transition(
        self,
        states: Union[torch.Tensor, Dict],
        actions: torch.Tensor,
        log_prob: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        timestep: int,
    ) -> None:
        """Record an environment transition in memory

        :param states: Observations/states of the environment used to make the decision
        :type states: torch.Tensor
        :param actions: Actions taken by the agent
        :type actions: torch.Tensor
        :param rewards: Instant rewards achieved by the current actions
        :type rewards: torch.Tensor
        :param next_states: Next observations/states of the environment
        :type next_states: torch.Tensor
        :param terminated: Signals to indicate that episodes have terminated
        :type terminated: torch.Tensor
        :param truncated: Signals to indicate that episodes have been truncated
        :type truncated: torch.Tensor
        :param infos: Additional information about the environment
        :type infos: Any type supported by the environment
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        # This just records info into self.tracking_data:
        # https://github.com/Toni-SM/skrl/blob/main/skrl/agents/torch/base.py
        # The parent class' method does not use the (next_)states in any way, so no need to handle any special cases
        # where the states returns from the envs are dicts or structured otherwise.
        # super().record_transition(
        #     states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps
        # )

        if self.memory is not None:

            self._current_next_states = next_states

            # reward shaping
            if self._rewards_shaper is not None:
                rewards = self._rewards_shaper(rewards, timestep, timestep)

            # compute values
            # When no state (or other) preprocessor is given to this class, this is the identity function
            with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                z = self.encoder(states)
                values = self.value.compute_value(z)
                # scale values back to true !
                # if torch.isnan(values).any():
                #     print(f"NaN/Inf detected in values 1")
                values = self._value_preprocessor(values, inverse=True)
                # if torch.isnan(values).any():
                #     print(f"NaN/Inf detected in values 2")

            # time-limit (truncation) boostrapping
            if self._time_limit_bootstrap:
                rewards += self._discount_factor * values * truncated

            # only store policy obs
            self.memory.add_samples(
                type="parallel",
                states={"policy": states["policy"]},
                actions=actions,
                rewards=rewards,
                next_states=None,
                terminated=terminated,
                truncated=truncated,
                log_prob=log_prob,
                values=values,
            )

    def _update(self) -> None:
        """Algorithm's main update step

        """

        def compute_gae(
            rewards: torch.Tensor,
            dones: torch.Tensor,
            values: torch.Tensor,
            next_values: torch.Tensor,
            discount_factor: float = 0.99,
            lambda_coefficient: float = 0.95,
        ) -> torch.Tensor:
            """Compute the Generalized Advantage Estimator (GAE)

            :param rewards: Rewards obtained by the agent
            :type rewards: torch.Tensor
            :param dones: Signals to indicate that episodes have ended
            :type dones: torch.Tensor
            :param values: Values obtained by the agent
            :type values: torch.Tensor
            :param next_values: Next values obtained by the agent
            :type next_values: torch.Tensor
            :param discount_factor: Discount factor
            :type discount_factor: float
            :param lambda_coefficient: Lambda coefficient
            :type lambda_coefficient: float

            :return: Generalized Advantage Estimator
            :rtype: torch.Tensor
            """
            advantage = 0
            advantages = torch.zeros_like(rewards)
            not_dones = dones.logical_not()
            memory_size = rewards.shape[0]

            # advantages computation
            for i in reversed(range(memory_size)):
                next_values = values[i + 1] if i < memory_size - 1 else last_values
                advantage = (
                    rewards[i]
                    - values[i]
                    + discount_factor * not_dones[i] * (next_values + lambda_coefficient * advantage)
                )
                advantages[i] = advantage
            # returns computation
            returns = advantages + values
            # normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            return returns, advantages
        
        # compute returns and advantages
        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            self.value.train(False)
            z = self.encoder(self._current_next_states)
            last_values = self.value.compute_value(z)
            self.value.train(True)
            last_values = self._value_preprocessor(last_values, inverse=True)
        last_values = last_values

        values = self.memory.get_tensor_by_name("values")

        returns, advantages = compute_gae(
            rewards=self.memory.get_tensor_by_name("rewards"),
            dones=self.memory.get_tensor_by_name("terminated") | self.memory.get_tensor_by_name("truncated"),
            values=values,
            next_values=last_values,
            discount_factor=self._discount_factor,
            lambda_coefficient=self._lambda,
        )


        # # Debug value components before loss calculation
        # if torch.isnan(returns).any():
        #     print("NaN in returns")
        # if torch.isnan(advantages).any():
        #     print("NaN in advantages")
        #     # Debug value components before loss calculation
        # if torch.isnan(values).any():
        #     print("NaN in values")
        # if torch.isnan(self._value_preprocessor(values)).any():
        #     print("NaN in value preprocessor")

        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)

        # sample mini-batches from memory
        sampled_batches = self.memory.sample_all(names=self._tensors_names, mini_batches=self._mini_batches)
        
        if self.auxiliary_task is not None:
            if self.auxiliary_task.use_same_memory:
                sampled_aux_batches = self.memory.sample_all(names=self._tensors_names, mini_batches=self._mini_batches)
            else:
                sampled_aux_batches = self.auxiliary_task.memory.sample_all(mini_batches=self._mini_batches)
            assert len(sampled_aux_batches) == len(sampled_batches) 
        else:
            sampled_aux_batches = None

        
        # turn policy and networks on
        self.policy.train()
        self.value.train()
        self.encoder.train()

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0
        cumulative_aux_loss = 0
        epoch_aux_loss = 0

        wandb_dict = {}

        # learning epochs
        for epoch in range(self._learning_epochs):
            kl_divergences = []

            # mini-batches loop
            for i, minibatch in enumerate(sampled_batches):
                if len(minibatch) == 6:
                    # sampled_states is a dict
                    (
                        sampled_states,
                        sampled_actions,
                        sampled_log_prob,
                        sampled_values,
                        sampled_returns,
                        sampled_advantages,
                    ) = minibatch  # noqa
                else:
                    raise ValueError("Check length of sampled states, should be dict")

                with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):

                    sampled_states = {"policy": sampled_states}

                    # self._check_instability(sampled_actions, "sampled_actions")
                    # self._check_instability(sampled_states["policy"]["prop"], "prop")
                    # self._check_instability(sampled_states["policy"]["tactile"], "tactile")

                    # self._check_instability(sampled_values, "sampled_values")
                    # self._check_instability(sampled_returns, "sampled_returns")
                    # self._check_instability(sampled_values, "sampled_values")
                    # self._check_instability(sampled_log_prob, "sampled_log_prob")
                    # self._check_instability(sampled_advantages, "sampled_advantages")

                    # torch autograd engine does not store gradients for leaf nodes by default
                    self.policy.log_std_parameter.retain_grad()

                    # update running mean/variance over all minibatches for just 1 epoch
                    z = self.encoder(sampled_states, train=not epoch)
                    _, next_log_prob, _ = self.policy.act(z, taken_actions=sampled_actions)

                    # compute approximate KL divergence
                    with torch.no_grad():
                        ratio = next_log_prob - sampled_log_prob
                        kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                        kl_divergences.append(kl_divergence)

                    # early stopping with KL divergence
                    if self._kl_threshold and kl_divergence > self._kl_threshold:
                        break

                    # compute entropy loss
                    if self._entropy_loss_scale:
                        entropy_loss = -self._entropy_loss_scale * self.policy.get_entropy(role="policy").mean()
                    else:
                        entropy_loss = 0

                    # compute policy loss
                    ratio = torch.exp(next_log_prob - sampled_log_prob)
                    surrogate = sampled_advantages * ratio
                    surrogate_clipped = sampled_advantages * torch.clip(
                        ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip
                    )

                    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                    # compute value loss
                    z = self.encoder(sampled_states)
                    predicted_values = self.value.compute_value(z)
                    # self._check_instability(predicted_values, "predicted_values")
                    # self._check_instability(z, "z")

                    # make sure predicted values have only moved a little bit for stability
                    if self._clip_predicted_values:
                        predicted_values = sampled_values + torch.clip(
                            predicted_values - sampled_values,
                            min=-self._value_clip,
                            max=self._value_clip,
                        )

                    value_loss = self._value_loss_scale * F.mse_loss(sampled_returns, predicted_values)

                    ## aux loss
                    sequential = False
                    if self.auxiliary_task is not None and sequential == False:
                        # aux_minibatch = (sampled_states["policy"],sampled_actions)
                        aux_minibatch_full = sampled_aux_batches[i]
                        aux_minibatch = (aux_minibatch_full[0], aux_minibatch_full[1])
                        # aux_minibatch = (sampled_states["policy"])
                        # aux_minibatch = (sampled_states,sampled_actions)
                        aux_loss, aux_info = self.auxiliary_task.compute_loss(aux_minibatch)
                        aux_loss *= self.auxiliary_task.aux_loss_weight

                        # self._check_instability(aux_loss, "aux_loss")

                        loss = (policy_loss + entropy_loss + value_loss + aux_loss)

                    else:
                        loss = (policy_loss + entropy_loss + value_loss)

                ## INDENT HERE
                # optimization step
                self.encoder_optimiser.zero_grad()
                self.policy_optimiser.zero_grad()
                self.value_optimiser.zero_grad()
                if self.auxiliary_task is not None:
                    self.auxiliary_task.optimiser.zero_grad()

                # check for NaN
                # Check loss before backward
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print(f"NaN/Inf detected in loss at epoch {epoch}, pruning trial")
                    self._check_instability(policy_loss, "policy_loss")
                    self._check_instability(value_loss, "value_loss")
                    self._check_instability(predicted_values, "predicted_values")
                    self._check_instability(sampled_actions, "sampled_actions")
                    self._check_instability(sampled_states["policy"]["prop"], "prop")
                    # self._check_instability(sampled_states["policy"]["tactile"], "tactile")
                    self._check_instability(sampled_values, "sampled_values")
                    self._check_instability(sampled_returns, "sampled_returns")
                    self._check_instability(sampled_values, "sampled_values")
                    self._check_instability(sampled_log_prob, "sampled_log_prob")
                    self._check_instability(sampled_advantages, "sampled_advantages")
                    
                    self.wandb_session.finish()
                    return True

                self.scaler.scale(loss).backward()
                
                # clip
                if self._grad_norm_clip > 0:
                    # self.scaler.unscale_(self.optimiser)
                    self.scaler.unscale_(self.encoder_optimiser)
                    self.scaler.unscale_(self.policy_optimiser)
                    self.scaler.unscale_(self.value_optimiser)
                    nn.utils.clip_grad_norm_(
                        itertools.chain(self.policy.parameters(), self.value.parameters(), self.encoder.parameters()),
                        self._grad_norm_clip,
                    )
                    # if self.auxiliary_task is not None:
                    #     self.scaler.unscale_(self.auxiliary_task.optimiser)
                    #     nn.utils.clip_grad_norm_(
                    #     itertools.chain(self.auxiliary_task.decoder.parameters()),
                    #     self._grad_norm_clip,
                    # )

                self.scaler.step(self.encoder_optimiser)
                self.scaler.step(self.policy_optimiser)
                self.scaler.step(self.value_optimiser)
                if self.auxiliary_task is not None:
                    self.scaler.step(self.auxiliary_task.optimiser)

                self.scaler.update()

                # update cumulative losses
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self.auxiliary_task is not None and sequential == False:
                    epoch_aux_loss += aux_loss.item()
                    cumulative_aux_loss += aux_loss.item()

                if self._entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()

            # update learning rate
            if self._learning_rate_scheduler:
                self.scheduler.step()

            if self.wandb_session is not None:
                wandb_dict["global_step"] = self.epoch_step
                wandb_dict["Loss / Aux epoch loss"] = epoch_aux_loss
                self.wandb_session.log(wandb_dict)
                if self.tb_writer is not None:
                    self.tb_writer.add_scalar("aux_epoch_loss", epoch_aux_loss, global_step=self.epoch_step)

            epoch_aux_loss = 0
            self.epoch_step += 1

        # wandb log
        if self.wandb_session is not None:
            wandb_dict["global_step"] = self.update_step
            avg_policy_loss = cumulative_policy_loss / (self._learning_epochs * self._mini_batches)
            avg_value_loss = cumulative_value_loss / (self._learning_epochs * self._mini_batches)
            wandb_dict["Loss / Policy loss"] = avg_policy_loss
            wandb_dict["Loss / Value loss"] = avg_value_loss
            if self.tb_writer is not None:
                self.tb_writer.add_scalar("policy_loss", avg_policy_loss, global_step=self.update_step)
                self.tb_writer.add_scalar("value_loss", avg_value_loss, global_step=self.update_step)

            if self.auxiliary_task is not None and sequential == False:
                avg_aux_loss = cumulative_aux_loss / (self._learning_epochs * self._mini_batches)
                wandb_dict["Loss / Aux loss"] = avg_aux_loss
                wandb_dict["Memory / size"] = len(self.memory)
                wandb_dict["Memory / memory_index"] = self.auxiliary_task.memory.memory_index
                wandb_dict["Memory / N_filled"] = int(self.auxiliary_task.memory.total_samples / self.auxiliary_task.memory.memory_size)
                wandb_dict["Memory / filled"] = float(self.auxiliary_task.memory.filled)
                wandb_dict["Loss / Entropy loss"] = cumulative_entropy_loss / (self._learning_epochs * self._mini_batches)
                if self.auxiliary_task.use_same_memory == False:
                    wandb_dict["Memory / mean_importance"] = float(self.auxiliary_task.memory.get_mean_importance())
                    wandb_dict["Memory / mean_return"] = float(self.auxiliary_task.memory.get_mean_return())
                    wandb_dict["Memory / mean_td_error"] = float(self.auxiliary_task.memory.get_mean_td_error())

                if self.tb_writer is not None:
                    self.tb_writer.add_scalar("aux_loss", avg_aux_loss, global_step=self.update_step)
                for k, v in aux_info.items():
                    # print(k,v)
                    wandb_dict[k] = v

                    if self.tb_writer is not None:
                        self.tb_writer.add_scalar(k, v, global_step=self.update_step)
                        
            if self._value_preprocessor is not None:
                wandb_dict["Scaling / input mean mean"] = self._value_preprocessor.running_mean_mean
                wandb_dict["Scaling / input mean median"] = self._value_preprocessor.running_mean_median
                wandb_dict["Scaling / input mean min"] = self._value_preprocessor.running_mean_min
                wandb_dict["Scaling / input mean max"] = self._value_preprocessor.running_mean_max
                wandb_dict["Scaling / input var mean"] = self._value_preprocessor.running_variance_mean
                wandb_dict["Scaling / input var median"] = self._value_preprocessor.running_variance_median
                wandb_dict["Scaling / input var min"] = self._value_preprocessor.running_variance_min
                wandb_dict["Scaling / input var max"] = self._value_preprocessor.running_variance_max

            wandb_dict["Policy / Standard deviation"] = self.policy.distribution(role="policy").stddev.mean().item()                
            self.wandb_session.log(wandb_dict)

        self.update_step += 1

        # turn policy and networks off for rollout collection
        self.policy.eval()
        self.value.eval()
        self.encoder.eval()
        # self.auxiliary_task.decoder.eval()

        # delete 
        # if self.auxiliary_task is not None and not self.auxiliary_task.use_same_memory:
        #     if self.auxiliary_task.memory_type == "prioritised":
        #         pass
        #     elif self.auxiliary_task.memory_type == "n_vanilla" and self.auxiliary_task.memory.filled:
        #         print("reseting n_vanilla  memory because its filled")
        #         self.auxiliary_task.memory.reset()
        #     elif self.auxiliary_task.memory_type == "reduced_vanilla":
        #         print("reseting reduced_vanilla memory")
        #         self.auxiliary_task.memory.reset()

        # no NaN encountered
        return False

    def _check_instability(self, x, name):
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"{name} is nan", torch.isnan(x).any())
            print(f"{name} is inf", torch.isinf(x).any())
        else:
            print(f"{name} is fine")
