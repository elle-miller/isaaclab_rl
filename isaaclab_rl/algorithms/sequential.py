import copy
import sys
import torch
import tqdm
from typing import List, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

import optuna 
from copy import deepcopy
from isaaclab_rl.algorithms import config, logger
from isaaclab_rl.algorithms.agent import Agent
from isaaclab_rl.wrappers.base_wrapper import Wrapper


# [start-config-dict-torch]
SEQUENTIAL_TRAINER_DEFAULT_CONFIG = {
    "timesteps": 100000,  # number of timesteps to train for
    "headless": False,  # whether to use headless mode (no rendering)
    "disable_progressbar": False,  # whether to disable the progressbar. If None, disable on non-TTY
    "close_environment_at_exit": False,  # whether to close the environment on normal program termination
    "environment_info": "episode",  # key used to get and log environment info
}
# [end-config-dict-torch]


class Trainer:
    def __init__(
        self,
        env: Wrapper,
        agents: Union[Agent, List[Agent]],
        agents_scope: Optional[List[int]] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        """Base class for trainers

        :param env: Environment to train on
        :type env: skrl.envs.wrappers.torch.Wrapper
        :param agents: Agents to train
        :type agents: Union[Agent, List[Agent]]
        :param agents_scope: Number of environments for each agent to train on (default: ``None``)
        :type agents_scope: tuple or list of int, optional
        :param cfg: Configuration dictionary (default: ``None``)
        :type cfg: dict, optional
        """
        self.cfg = cfg if cfg is not None else {}
        self.env = env
        self.agents = agents
        self.agents_scope = agents_scope if agents_scope is not None else []

        # get configuration
        self.timesteps = self.cfg.get("timesteps", 0)
        self.headless = self.cfg.get("headless", False)
        self.disable_progressbar = self.cfg.get("disable_progressbar", False)
        self.close_environment_at_exit = self.cfg.get("close_environment_at_exit", True)
        self.environment_info = self.cfg.get("environment_info", "episode")

        self.initial_timestep = 0

        # Using this trainer in an alternating fashion (e.g., .train() -> .eval() -> .train()) will restart the
        # env on each call to .train(). This is an issue if we are using the SKRL memory class, as it is not
        # aware of manual restarts. This means we will be storing a trajectory that will randomly have resets
        # *without* a DONE flag. This can cause learning instabilities. Therefore, ONLY CALL .reset() ONCE!
        # self.started_already = False

        # setup agents
        self.num_simultaneous_agents = 0
        self._setup_agents()

        # register environment closing if configured
        if self.close_environment_at_exit:

            @atexit.register
            def close_env():
                logger.info("WARNING SHOULD NEVER BE HERE Closing environment")
                # self.env.close()
                logger.info("Environment closed")

        # update trainer configuration to avoid duplicated info/data in distributed runs
        if config.torch.is_distributed:
            if config.torch.rank:
                self.disable_progressbar = True

    def __str__(self) -> str:
        """Generate a string representation of the trainer

        :return: Representation of the trainer as string
        :rtype: str
        """
        string = f"Trainer: {self}"
        string += f"\n  |-- Number of parallelizable environments: {self.env.num_envs}"
        string += f"\n  |-- Number of simultaneous agents: {self.num_simultaneous_agents}"
        string += "\n  |-- Agents and scopes:"
        if self.num_simultaneous_agents > 1:
            for agent, scope in zip(self.agents, self.agents_scope):
                string += f"\n  |     |-- agent: {type(agent)}"
                string += f"\n  |     |     |-- scope: {scope[1] - scope[0]} environments ({scope[0]}:{scope[1]})"
        else:
            string += f"\n  |     |-- agent: {type(self.agents)}"
            string += f"\n  |     |     |-- scope: {self.env.num_envs} environment(s)"
        return string

    def _setup_agents(self) -> None:
        """Setup agents for training

        :raises ValueError: Invalid setup
        """
        # validate agents and their scopes
        if type(self.agents) in [tuple, list]:
            # single agent
            if len(self.agents) == 1:
                self.num_simultaneous_agents = 1
                self.agents = self.agents[0]
                self.agents_scope = [1]
            # parallel agents
            elif len(self.agents) > 1:
                self.num_simultaneous_agents = len(self.agents)
                # check scopes
                if not len(self.agents_scope):
                    logger.warning("The agents' scopes are empty, they will be generated as equal as possible")
                    self.agents_scope = [int(self.env.num_envs / len(self.agents))] * len(self.agents)
                    if sum(self.agents_scope):
                        self.agents_scope[-1] += self.env.num_envs - sum(self.agents_scope)
                    else:
                        raise ValueError(
                            f"The number of agents ({len(self.agents)}) is greater than the number of parallelizable environments ({self.env.num_envs})"
                        )
                elif len(self.agents_scope) != len(self.agents):
                    raise ValueError(
                        f"The number of agents ({len(self.agents)}) doesn't match the number of scopes ({len(self.agents_scope)})"
                    )
                elif sum(self.agents_scope) != self.env.num_envs:
                    raise ValueError(
                        f"The scopes ({sum(self.agents_scope)}) don't cover the number of parallelizable environments ({self.env.num_envs})"
                    )
                # generate agents' scopes
                index = 0
                for i in range(len(self.agents_scope)):
                    index += self.agents_scope[i]
                    self.agents_scope[i] = (index - self.agents_scope[i], index)
            else:
                raise ValueError("A list of agents is expected")
        else:
            self.num_simultaneous_agents = 1

    def train(self) -> None:
        """Train the agents

        :raises NotImplementedError: Not implemented
        """
        raise NotImplementedError

    def eval(self) -> None:
        """Evaluate the agents

        :raises NotImplementedError: Not implemented
        """
        raise NotImplementedError


class SequentialTrainer(Trainer):
    def __init__(
        self,
        env: Wrapper,
        agents: Union[Agent, List[Agent]],
        agents_scope: Optional[List[int]] = None,
        cfg: Optional[dict] = None,
        num_eval_envs = 1,
        auxiliary_task = None,
    ) -> None:
        """Sequential trainer

        Train agents sequentially (i.e., one after the other in each interaction with the environment)

        :param env: Environment to train on
        :type env: skrl.envs.wrappers.torch.Wrapper
        :param agents: Agents to train
        :type agents: Union[Agent, List[Agent]]
        :param agents_scope: Number of environments for each agent to train on (default: ``None``)
        :type agents_scope: tuple or list of int, optional
        :param cfg: Configuration dictionary (default: ``None``).
                    See SEQUENTIAL_TRAINER_DEFAULT_CONFIG for default values
        :type cfg: dict, optional
        """
        _cfg = copy.deepcopy(SEQUENTIAL_TRAINER_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        agents_scope = agents_scope if agents_scope is not None else []
        super().__init__(env=env, agents=agents, agents_scope=agents_scope, cfg=_cfg)

        # init agents
        self.agents.init(trainer_cfg=self.cfg)

        # this is the timesteps per environment
        self.training_timestep = 0

        # global steps accumulates over all environments i.e. global step = num envs * training steps
        self.global_step = 0
        self.num_envs = env.num_envs
        self.num_eval_envs = num_eval_envs

        # this is the eval equivalent for training_timestep, just for wandb tracking
        self.eval_timestep = 0

        self.auxiliary_task = auxiliary_task

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.encoder = self.agents.encoder

    def fill_physics_buffer(self, wandb_session, tb_writer=None) -> None:
        """Train the agents sequentially

        Fill up the buffer
        """

        # HARD reset of all environments to begin evaluation
        states, infos = self.env.reset(hard=True)

        # Reset physics memory
        self.auxiliary_task.memory.reset()

        explore_timesteps = 10000
        rollout = 0
        rollout_length = 32
        for timestep in range(explore_timesteps):
 
            # update global step
            self.global_explore_step = timestep * self.num_envs

            # create new wandb dict
            self.wandb_timestep_dict = {}
            self.wandb_timestep_dict["global_step"] = self.global_explore_step

            # compute actions
            with torch.no_grad():
                z = self.encoder(states)

                # TODO: replace this with adverserial/curious policy to pick action of highest uncertainty 
                # actions, log_prob, outputs = self.agents.random_act(z)
                actions, log_prob, outputs = self.agents.policy.act(z)

                # step the environments
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)

                # render scene
                if not self.headless:
                    self.env.render()

                # the logic of adding samples is left to the aux task
                if not self.auxiliary_task.memory.filled:
                    self.auxiliary_task.add_samples(
                        states=deepcopy(states),
                        actions=deepcopy(actions),
                        next_states=deepcopy(next_states),
                        terminated=deepcopy(terminated),
                        truncated=deepcopy(truncated),
                    )

                states = next_states

            # begin grad! 
            rollout += 1
            # if not rollout % rollout_length:
            if self.auxiliary_task.memory.filled:
                print(f"filled after {timestep}, updating forever")
                for i in range(10000):
                    self.auxiliary_task._update()
                    self.agents.write_checkpoint(i, timestep=i, timesteps=None)
                rollout = 0

            # if self.auxiliary_task.memory.filled:
            #     self.auxiliary_task._update()
            #     return
                    
            # if wandb_session is not None:
            #     wandb_session.log(self.wandb_timestep_dict)



    def train(self, wandb_session, tb_writer=None, record=False, play=False, trial=None) -> None:
        """Train the agents sequentially

        This method executes the following steps in loop:

        - Pre-interaction (sequentially)
        - Compute actions (sequentially)
        - Interact with the environments
        - Render scene
        - Record transitions (sequentially)
        - Post-interaction (sequentially)
        - Reset environments
        """
        # set running mode
        assert self.num_simultaneous_agents == 1, "This method is not allowed for simultaneous agents"
        assert self.env.num_agents == 1, "This method is not allowed for multi-agents"

        # HARD reset of all environments to begin evaluation
        states, infos = self.env.reset(hard=True)

        # Resetting here helps with .train()->.eval()->.train() The first .train() rollout could be interrupted by
        # the call to .eval(). This interruption is likely not recorded in the memory, so the training stage
        # may compute information across trajectories, which is not ideal.
        # We also need to reset the agent's "_rollout" attribute, as this determines when the agent is actually
        # updated. Resetting it here ensures that each agent update happens with the hyperparam-specified
        # frequency.
        self.agents.memory.reset()

        # get ready
        num_envs = self.env.num_envs

        self.returns_dict, self.infos_dict, self.mask, self.term_mask, self.trunc_mask = self.get_empty_return_dicts(infos)
        
        ep_length = self.env.env.unwrapped.max_episode_length - 1
        images = []
        # save_gif = True if "pixels" in env_cfg.obs_list else False


        # metrics where we only care about mean over whole episode in context of training
        wandb_episode_dict = {}
        wandb_episode_dict["global_step"] = self.global_step

        env_id = 0
        best_return = 0

        # counter variable for which step we are on
        rollout = 0
        self.share_memory = False

        self.rl_update = 0

        # print(train_start, train_pause)

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps),
            disable=self.disable_progressbar,
            file=sys.stdout,
        ):
            
            # update global step
            self.global_step = timestep * self.num_envs

            # create new wandb dict
            self.wandb_timestep_dict = {}
            self.wandb_timestep_dict["global_step"] = self.global_step

            # compute actions
            with torch.no_grad():
                z = self.encoder(states)
                
                # For evaluation environments
                if self.num_eval_envs > 0:
                    eval_z = z[:self.num_eval_envs]
                    eval_actions, _, _ = self.agents.policy.act(eval_z, deterministic=True)
                    # eval_actions, _, _ = self.agents.policy.act(eval_z)

                # For training environments
                train_z = z[self.num_eval_envs:]
                train_actions, train_log_prob, outputs = self.agents.policy.act(train_z)
                
                # Combine actions
                actions = torch.zeros((self.num_envs, train_actions.shape[1])).to(self.device)
                actions[:self.num_eval_envs] = eval_actions
                actions[self.num_eval_envs:] = train_actions

                # step the environments
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)

                # render scene
                if not self.headless:
                    self.env.render()

                self.save_transition_to_memory(states, actions, train_log_prob, rewards, next_states, terminated, truncated, infos)

            # begin grad! 
            rollout += 1
            if not rollout % self.agents._rollouts:
                
                nan_encountered = self.agents._update()
                if nan_encountered:
                    return float("nan"), True

                self.rl_update += 1
                rollout = 0

            states = next_states
                
            # reset environments
            # the eval episodes get manually reset every ep_length
            if timestep > 0 and (timestep % ep_length == 0) and self.num_eval_envs > 0:

                # take counter item, mean across eval envs
                for k, v in infos["counters"].items():
                    wandb_episode_dict[f"Eval episode counters / {k}"] = v[:self.num_eval_envs].mean().cpu()
                    if tb_writer is not None:
                        tb_writer.add_scalar(f"{k}", v[:self.num_eval_envs].mean().cpu(), global_step=self.global_step)
                        
                # reset state of scene
                # manually cause a reset by flagging done ?
                indices = torch.arange(self.num_eval_envs, dtype=torch.int64, device=self.device)
                obs, _ = self.env.reset(env_ids=indices)

                # # TODO: CHECK IF FRAME STACK HASN'T MESSED UP NON-EVAL ENV OBS
                # update states TODO CHECK IF THIS IS LEGIT
                states = obs

                # metrics where we only care about mean over whole episode in context of training
                # update the episode dict
                for k, v in self.returns_dict.items():
                    wandb_episode_dict[f"Eval episode returns / {k}"] = v.mean().cpu()

                for k, v in self.infos_dict.items():
                    wandb_episode_dict[f"Eval episode info / {k}"] = v.mean().cpu()
                    if tb_writer is not None:
                        tb_writer.add_scalar(f"{k}", v.mean().cpu(), global_step=self.global_step)

                if wandb_session is not None:
                    wandb_session.log(wandb_episode_dict)

                wandb_episode_dict = {}
                wandb_episode_dict["global_step"] = self.global_step

                mean_eval_return = self.returns_dict["returns"].mean().item()
                print(f"{self.global_step/1e6} M: {mean_eval_return}\n")

                # write checkpoints
                if not play:
                    self.agents.write_checkpoint(mean_eval_return, timestep=self.global_step, timesteps=None)
                if tb_writer is not None:
                    tb_writer.add_scalar(f"mean_eval_return", mean_eval_return, global_step=self.global_step)

                self.returns_dict, self.infos_dict, self.mask, self.term_mask, self.trunc_mask = self.get_empty_return_dicts(infos)

                # sweep stuff
                if trial is not None:
                    if mean_eval_return > best_return:
                        best_return = mean_eval_return
                    trial.report(mean_eval_return, self.global_step)
                    if trial.should_prune():
                        return best_return, True

        return best_return, False
    
    def replace_eval_obs(self, obs, new_obs, num_eval_envs):
        for type in obs:
            for k in type:
                obs[type][k][:num_eval_envs] = new_obs[type][k][:num_eval_envs]
        return obs
    
    def save_transition_to_memory(self, states, actions, train_log_prob, rewards, next_states, terminated, truncated, infos):

        # AUXILIARY MEMORY
        if self.auxiliary_task is not None and self.auxiliary_task.use_same_memory is False:
            # doing a deep copy because i observed changing the tensors in aux task messed up ppo
            # if not self.auxiliary_task.memory.filled:
            # deep copy doesn't seem to make much of a memory difference, so leaving in
            if isinstance(self.auxiliary_task, ForwardDynamics):
                self.auxiliary_task.add_samples(
                    states=deepcopy(states),
                    actions=deepcopy(actions),
                    rewards=deepcopy(rewards),
                    done=deepcopy(terminated|truncated),
                )
            elif isinstance(self.auxiliary_task, Reconstruction):
                self.auxiliary_task.add_samples(
                    states=states,
                )
            else:
                raise ValueError

        # then mess up for PPO training
        train_states = self.get_last_n_obs(states, self.num_eval_envs)
        train_next_states = self.get_last_n_obs(next_states, self.num_eval_envs)
        train_actions = actions[self.num_eval_envs:, :]
        train_rewards = rewards[self.num_eval_envs:, :]
        train_terminated = terminated[self.num_eval_envs:, :]
        train_truncated = truncated[self.num_eval_envs:, :]

        # eval_states = self.get_last_n_obs(states, self.num_eval_envs)
        # eval_next_states = self.get_last_n_obs(next_states, self.num_eval_envs)
        # eval_actions = actions[:self.num_eval_envs, :]
        eval_rewards = rewards[:self.num_eval_envs, :]
        eval_terminated = terminated[:self.num_eval_envs, :]
        eval_truncated = truncated[:self.num_eval_envs, :]
        mask_update = 1 - torch.logical_or(eval_terminated, eval_truncated).float()

        # these are metrics added to self.extras["log"] in the environment at each timestep
        if "log" in infos:
            for k, v in infos["log"].items():
                # timestep logging
                self.wandb_timestep_dict[f"Eval timestep / {k}"] = v[:self.num_eval_envs].cpu()
                self.infos_dict[k] += v[:self.num_eval_envs].mean() * self.mask

        # update eval dicts
        self.returns_dict["unmasked_returns"] += eval_rewards
        self.returns_dict["returns"] += eval_rewards * self.mask
        self.returns_dict["steps_to_term"] += self.term_mask[:self.num_eval_envs]
        self.returns_dict["steps_to_trunc"] += self.trunc_mask[:self.num_eval_envs]
        self.mask *= mask_update
        self.term_mask *= 1 - terminated[:self.num_eval_envs].float()
        self.trunc_mask *= 1 - truncated[:self.num_eval_envs].float()
        
        # record to PPO memory
        self.agents.record_transition(
            states=train_states,
            actions=train_actions,
            log_prob=train_log_prob,
            rewards=train_rewards,
            next_states=train_next_states,
            terminated=train_terminated,
            truncated=train_truncated,
            timestep=self.global_step,
        ) 


    def get_last_n_obs(self, obs, n):
        result = {}
        for k, v in obs.items():
            result[k] = {}
            for key, value in obs[k].items(): 
                    result[k][key] = value[:][n:]
        return result


    def get_empty_return_dicts(self, infos):
        returns = {
            "returns": None,
            "unmasked_returns": None,
            "steps_to_term": None,
            "steps_to_trunc": None,
        }

        returns_dict = {k: torch.zeros(size=(self.num_eval_envs, 1), device=self.device) for k in returns.keys()}
        infos_dict = {k: torch.zeros(size=(self.num_eval_envs, 1), device=self.device) for k in infos["log"].keys()}

        mask = torch.Tensor([[1] for _ in range(self.num_eval_envs)]).to(self.device)
        term_mask = torch.Tensor([[1] for _ in range(self.num_eval_envs)]).to(self.device)
        trunc_mask = torch.Tensor([[1] for _ in range(self.num_eval_envs)]).to(self.device)
        return returns_dict, infos_dict, mask, term_mask, trunc_mask
    

    def eval(self, wandb_session=None, tb_writer=None, record=False, play=False) -> None:
        """Evaluate the agents sequentially

        This method executes the following steps in loop:

        - Compute actions (sequentially)
        - Interact with the environments
        - Render scene
        - Reset environments
        """


        assert self.num_simultaneous_agents == 1, "This method is not allowed for simultaneous agents"
        assert self.env.num_agents == 1, "This method is not allowed for multi-agents"

        # Hard reset all environments
        states, infos = self.env.reset(hard=True)

        # get ready
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        num_envs = self.env.num_envs
        returns = {
            "returns": None,
            "unmasked_returns": None,
            "steps_to_term": None,
            "steps_to_trunc": None,
        }
        returns_dict = {k: torch.zeros(size=(num_envs, 1), device=device) for k in returns.keys()}
        infos_dict = {}
        mask = torch.Tensor([[1] for _ in range(num_envs)]).to(device)
        term_mask = torch.Tensor([[1] for _ in range(num_envs)]).to(device)
        trunc_mask = torch.Tensor([[1] for _ in range(num_envs)]).to(device)
        ep_length = self.env.env.unwrapped.max_episode_length - 1
        images = []

        # metrics where we only care about mean over whole episode in context of training
        wandb_episode_dict = {}
        wandb_episode_dict["global_step"] = self.global_step

        self.eval_timestep = self.global_step

        env_id = 0

        # run eval episode
        for timestep in tqdm.tqdm(
            range(self.initial_timestep, ep_length),
            disable=self.disable_progressbar,
            file=sys.stdout,
        ):
            # log some metrics each timestep (e.g. to see forces over time in one episode)
            wandb_timestep_dict = {}
            wandb_timestep_dict["global_step"] = self.eval_timestep

            # compute actions
            with torch.no_grad():
                z = self.encoder(states)
                actions, _, _ = self.agents.policy.act(z, deterministic=True)
                # actions = self.agents.policy.act(z)[0]

                # step the environments
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)

                if wandb_session is not None:
                    self.log_sar(wandb_timestep_dict, states["policy"], actions, rewards)

                mask_update = 1 - torch.logical_or(terminated, truncated).float()

                # these are metrics added to self.extras["log"] in the environment at each timestep
                if "log" in infos:
                    for k, v in infos["log"].items():
                        # timestep logging
                        wandb_timestep_dict[f"Eval timestep / {k}"] = v[env_id].cpu()

                        # if k == "left_x": # and v[env_id].cpu() > 0:
                        #     print("left x", v[env_id].cpu())

                        # if k == "forces_rotatedx":
                        #     print("forces", v[env_id].cpu())

                        # episode logging
                        infos_dict[k] = torch.zeros(size=(num_envs, 1), device=device)
                        infos_dict[k] += v.mean() * mask

                # update
                returns_dict["unmasked_returns"] += rewards
                returns_dict["returns"] += rewards * mask
                returns_dict["steps_to_term"] += term_mask
                returns_dict["steps_to_trunc"] += trunc_mask
                mask *= mask_update
                term_mask *= 1 - terminated.float()
                trunc_mask *= 1 - truncated.float()

                if record:
                    # The 1st [:] is to "activate" the LazyTensor
                    images.append(next_states["policy"]["pixels"][:][0].unsqueeze(0))

                # render scene
                if not self.headless:
                    self.env.render()

                # reset environments
                states = next_states

                if wandb_session is not None:
                    wandb_session.log(wandb_timestep_dict)

                self.eval_timestep += 1

        # exit()
        return returns_dict, images

    def log_sar(self, wandb_timestep_dict, states, actions, rewards):
        if type(states) == dict:
            for k in sorted(states.keys()):
                if k != "pixels":
                    wandb_timestep_dict[f"Eval debug / states = {k}"] = states[k][:].mean().cpu()
        else:
            wandb_timestep_dict[f"Eval debug / states"] = states.mean().cpu()

        wandb_timestep_dict[f"Eval debug / actions"] = actions.mean().cpu()
        wandb_timestep_dict[f"Eval debug / rewards"] = rewards.mean().cpu()

    def plot_value(self, states=None, image=None, t=0):

        if states is None:
            states, infos = self.env.reset(hard=True)

        # modify styffh
        # states = states["policy"]

        # Define the range for the two inputs of interest
        tactile_left_range = np.linspace(0, 1, 100)  # Range for input 1
        tactile_right_range = np.linspace(0, 1, 100)  # Range for input 2

        # Create a grid of states
        input1_grid, input2_grid = np.meshgrid(tactile_left_range, tactile_right_range)

        # Initialize the value function grid
        value_grid = np.zeros_like(input1_grid)

        # Evaluate the value function over the grid
        with torch.no_grad():  # Disable gradient computation for efficiency
            for i in range(input1_grid.shape[0]):
                for j in range(input1_grid.shape[1]):
                    # Create the full state vector
                    states["policy"]["tactile"] = torch.tensor([[input1_grid[i, j], input2_grid[i, j]]], dtype=torch.float32).to(self.agents.device)
                    # Query the neural network for the value
                    z = self.agents.fusion_network(states["policy"])
                    value = self.agents.value.net(z).item()
                    value_grid[i, j] = value

        # Plot the value function
        plt.figure(figsize=(10, 8))
        contour = plt.contourf(input1_grid, input2_grid, value_grid, levels=50, cmap='viridis')
        # plt.colorbar(contour, label='Value Function')
        plt.colorbar(contour, boundaries=np.linspace(-1,2,100)) 

        plt.xlabel('Left tactile input')
        plt.ylabel('Right tactile input')
        plt.title(f'V(s) @ t={t}')
        directory = "/home/elle/code/external/IsaacLab/isaaclab_rl/results/plots/value"
        file = os.path.join(directory, f"value_{t}.png")
        plt.savefig(file)

        plt.clf()


        # save image
        image = states["aux"]["pixels"][:][0]
        image = image.detach().cpu()


        # image size is [3, 84, 84]
        if np.argmin(image.shape) == 2:
            permute_order = (0, 1, 2)
        else:
            permute_order = (1, 2, 0)

        if image.dtype is torch.float32:
            image *= 255

        img = np.array(image.permute(permute_order)[:, :, :3]).astype(np.uint8)
        file = os.path.join(directory, f"robot_{t}.png")

        im = Image.fromarray(img)
        im.save(file)
        plt.savefig(file)
        
        # plt.show()

