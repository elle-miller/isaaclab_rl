from abc import ABC, abstractmethod
import kornia


import torch
import torch.nn as nn

import kornia
import numpy as np
from isaaclab_rl.algorithms.memories import Memory
import gc

import torch.nn as nn

from torch.optim.lr_scheduler import LambdaLR, ExponentialLR, ReduceLROnPlateau
import kornia.losses
import numpy as np
from copy import deepcopy


class AuxiliaryTask(ABC):
    """
    Abstract base class for all auxiliary tasks.
    Each subclass should implement its own `compute_loss`.
    """

    def __init__(self, aux_task_cfg, rl_rollout, rl_memory, encoder, value, value_preprocessor, env, env_cfg, writer):
        # hparams
        self.aux_loss_weight = aux_task_cfg["loss_weight"]
        self.lr = aux_task_cfg["learning_rate"]
        self.mini_batches = aux_task_cfg["mini_batches"]
        self.learning_epochs = aux_task_cfg["learning_epochs"]
        self.rl_per_aux = aux_task_cfg["rl_per_aux"]
        self.clip_grad = aux_task_cfg["clip_grad"]
        self.tactile_only = aux_task_cfg["tactile_only"]
        self.metric = None # aux_task_cfg["metric"]

        self.last_n_obs = aux_task_cfg["last_n_obs"]
        self.n_f = aux_task_cfg["n_f"]

        self.memory_type = aux_task_cfg["memory_type"]
        # use PPO memory, aka no memory
        if self.memory_type == "vanilla":
            self.use_same_memory = True
            self.memory = rl_memory
            print("*******************AUX USING SAME MEMORY ")
        # otherwise create my own memory

        else:
            self.use_same_memory = False

        self.memory_size = aux_task_cfg["memory_size"]

        print("MEM TYPES AND SIZES", self.memory_type, self.memory_size)
        if self.memory_size == 0:
            if self.memory_type == "n_vanilla" or self.memory_type == "prioritised":
                print("Vanilla memory with n=", self.rl_per_aux)
                self.memory_size = self.rl_per_aux * rl_rollout * env.num_envs
            else:
                self.memory_size = rl_rollout * env.num_envs
            print("Updated AUX memory size to", self.memory_size)

        # sample indices randomly instead of sequentially when generating minibatches
        self.random_sample = True

        self.rl_rollout = rl_rollout
        self.env = env
        self.env_cfg = env_cfg
        self.num_eval_envs = env_cfg.num_eval_envs
        self.num_training_envs = env.num_envs - self.num_eval_envs
        self.device = env.device
        self.wandb_session = writer.wandb_session
        self.tb_writer = writer.tb_writer

        # optimise everything together, could separate in future
        self.encoder = encoder
        self.value = value
        self.value_preprocessor = value_preprocessor
        self.z_dim = self.encoder.num_outputs
        self.action_dim = self.env.action_space.shape[0]

        if aux_task_cfg["augment"] == True:
            img_dim = self.encoder.img_dim
            self.augmentations = nn.Sequential(
                nn.ReplicationPad2d(4), kornia.augmentation.RandomCrop((img_dim, img_dim)),
                # kornia.augmentation.RandomGaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0), p=0.5)
            )
        else:
            self.augmentations = None

        # default tensor sampling is states + actoins
        # TODO; just make states and only add actions for dynamics
        self._aux_tensors_names = []
        for type_k in sorted(self.env.observation_space.keys()):
            for k, v in self.env.observation_space[type_k].items():
                self._aux_tensors_names.append(k)
        self._aux_tensors_names = self._aux_tensors_names + ["actions"]

        # target network params
        self.encoder_per_target_update = 1
        self.tau = 0.01

        self.update_step = 0
        self.minibatch_step = 0

        # stuff
        self.oracle = False
        self.percent_learnable = 1

        # set up automatic mixed precision
        self._mixed_precision = True
        self._device_type = torch.device(self.device).type
        self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self._mixed_precision)


    def _post_init(self):
        # we need to create networks before this
        self.optimisable_networks = self.set_optimisable_networks()

        self.optimiser = torch.optim.Adam( 
            [param for net in self.optimisable_networks for param in net.parameters()], 
            lr=self.lr,
        )
        
        # defaults, can update in local tasks
        self.scheduler = ReduceLROnPlateau(self.optimiser, 'min', factor=0.9, min_lr=1e-5)
        self.criterion = nn.MSELoss()

        if not self.use_same_memory:
            print("*******************AUX USING OWN MEMORY ")
            self.memory = self.create_memory()
            self.memory.reset()
            # self.create_memory_tensors()

    @abstractmethod
    def set_optimisable_networks(self):
        pass

    @abstractmethod
    def create_memory(self):
        pass

    @abstractmethod
    def sample_minibatches(self):
        pass

    @abstractmethod
    def compute_loss(self):
        pass

    def _update(self):
        """
        This is used by all tasks 
        1. sample minibatches from the memory
        2. compute loss on minibatch
        3. back up with loss
        """

        # set optimisable networks to train
        self.set_networks_mode(self.optimisable_networks, "train")

        cumulative_aux_loss = 0

        # this loops through different memories if they exist
        batches = self.sample_minibatches()
        for batch in batches:
            sampled_batches = batch
            for epoch in range(self.learning_epochs):
                for i, minibatch in enumerate(sampled_batches):

                    with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                        loss, info = self.compute_loss(minibatch)

                        # update networks
                        loss *= self.aux_loss_weight

                    self.optimiser.zero_grad()
                    
                    # loss.backward()
                    self.scaler.scale(loss).backward()
                
                    self.scaler.step(self.optimiser)
                    self.scaler.update()

                    cumulative_aux_loss += loss
                
                    # Then call garbage collection
                    # del minibatch
                    # gc.collect()
                    # torch.cuda.empty_cache()
                    # wandb log
                    if self.wandb_session is not None:
                        wandb_dict = {}
                        wandb_dict["global_step"] = self.minibatch_step
                        wandb_dict["Loss / minibatch"] = loss.item()
                        self.wandb_session.log(wandb_dict)

       
        self.update_step += 1

        # step scheduler
        self.scheduler.step(cumulative_aux_loss)

        av_aux_loss = cumulative_aux_loss / (self.learning_epochs * self.mini_batches)

        print(f"Auxiliary loss: {av_aux_loss}")
        print(f"Auxiliary lr: {self.scheduler.get_last_lr()[0]}")

        # wandb log
        if self.wandb_session is not None:
            wandb_dict = {}
            wandb_dict["global_step"] = self.update_step
            wandb_dict["Loss / Aux LR"] = self.scheduler.get_last_lr()[0]
            wandb_dict["Loss / Auxiliary loss"] = av_aux_loss
            wandb_dict["Memory / size"] = len(self.memory)
            wandb_dict["Memory / memory_index"] = self.memory.memory_index
            wandb_dict["Memory / N_filled"] = int(self.memory.total_samples / self.memory.memory_size)
            wandb_dict["Memory / filled"] = float(self.memory.filled)
            # wandb_dict["Memory / learnable %"] = float(self.memory.filled)
            # wandb_dict["Memory / mean_importance"] = float(self.memory.get_mean_importance())


            for k, v in info.items():
                wandb_dict[k] = v

            self.wandb_session.log(wandb_dict)

        # set optimisable networks to train
        self.set_networks_mode(self.optimisable_networks, "eval")

    def create_sequential_memory(self, size=10000):
        """
        This is to store more discrete events like contacts
        """
        return Memory(
                memory_size=int(size),
                num_envs=1,
                device=self.device,
                env_cfg=self.env_cfg,
        )
    
    def create_parallel_memory(self):
        """
        This collects every transition by each env for N rollouts
        """
        return Memory(
                memory_size=self.rl_per_aux*self.rl_rollout,
                num_envs=self.num_training_envs,
                device=self.device,
                env_cfg=self.env_cfg,
        )
    
    def create_memory_tensors(self):
        """
        Create observation and action tensors for the aux memory
        """

        observation_names = []
        # outer loop of observation space (policy, aux)
        for type_k in sorted(self.env.observation_space.keys()):
            for k, v in self.env.observation_space[type_k].items():
                # create next states for the forward_dynamics
                print(f"AuxiliaryTask: {k}: {type_k} tensor size {v.shape}")
                self.memory.create_tensor(
                    name=k,
                    size=v.shape,
                    dtype=torch.uint8 if k == "pixels" else torch.float32,
                )
                observation_names.append(k)
            
        self.memory.create_tensor(name="actions", size=self.env.action_space, dtype=torch.float32)


    def add_samples(self, states, actions, next_states, terminated, truncated):
        """
        Add samples to dedicated aux memory
        Re-implement this if you don't need all of these tensors

        only allowed for not shared memory, can't mess up the rl one
        
        """
        if not self.use_same_memory:
            self.memory.add_samples(
                states=states,
                actions=actions,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated
            )

    def soft_update_params(self, net, target_net, tau=0.05):
        """
        Slowly update target network to avoid moving target problem
        """
        if (self.update_step % self.encoder_per_target_update) == 0:
            for param, target_param in zip(net.parameters(), target_net.parameters()):
                target_param.data.copy_(tau * param.data +
                                        (1 - tau) * target_param.data)
            

    def separate_memory_tensors(self, sampled_states):
        # separate policy tensors from aux tensors
        sampled_states_dict = {}
        for k in sorted(self.env.observation_space.keys()):     # loops through "policy", "aux"
            sampled_states_dict[k] = {}
            for obs_k in sorted(self.env.observation_space[k].keys()):  # loops through obs keys
                # print(f"adding {obs_k} to {k}")
                sampled_states_dict[k][obs_k] = sampled_states[obs_k]

        if any('next_' in s for s in sampled_states.keys()):

            next_sampled_states_dict = {}
            for k in sorted(self.env.observation_space.keys()):     # loops through "policy", "aux"
                next_sampled_states_dict[k] = {}
                # these will be "next_prop", but now we just wanna save as "prop"
                for obs_k in sorted(self.env.observation_space[k].keys()):  # loops through obs keys
                    next_name = "next_" + obs_k
                    if next_name in sampled_states.keys():
                        next_sampled_states_dict[k][obs_k] = sampled_states[next_name]

            return sampled_states_dict, next_sampled_states_dict

        return sampled_states_dict, None
    
    # Toggling between train and eval modes
    def set_networks_mode(self, networks, mode='train'):
        for net in networks:
            if mode == 'train':
                net.train()
            else:
                net.eval()

    def evaluate_binary_predictions(self, predictions, targets, step=0, threshold=0.5):
        
        # Convert to binary predictions using threshold
        binary_preds = (predictions >= threshold).float()
        
        # Count total correct predictions
        correct = (binary_preds == targets).float().sum().item()
        total = targets.numel()
        
        # Count correct 1s and 0s separately
        true_positives = ((binary_preds == 1) & (targets == 1)).float().sum().item()
        false_positives = ((binary_preds == 1) & (targets == 0)).float().sum().item()
        true_negatives = ((binary_preds == 0) & (targets == 0)).float().sum().item()
        false_negatives = ((binary_preds == 0) & (targets == 1)).float().sum().item()

        # Count total 1s and 0s
        total_positives = (targets == 1).float().sum().item()
        total_negatives = (targets == 0).float().sum().item()
        
        # Calculate metrics
        accuracy = correct / total
        precision = true_positives / max(1, (binary_preds == 1).float().sum().item())
        recall = true_positives / max(1, total_positives)
        f1 = 2 * precision * recall / max(1e-8, precision + recall)

        self.minibatch_step += 1

        tp_rate = true_positives / max(1, total_positives)
        fp_rate = false_positives / max(1, total_positives)
        tn_rate = true_negatives / max(1, total_negatives)
        fn_rate = false_negatives / max(1, total_negatives)

        # if self.tb_writer is not None:
        #     self.tb_writer.add_scalar(f"accuracy@t={step}", accuracy, global_step=self.minibatch_step)
        #     self.tb_writer.add_scalar(f"precision@t={step}", precision, global_step=self.minibatch_step)
        #     self.tb_writer.add_scalar(f"recall@t={step}", recall, global_step=self.minibatch_step)
        #     self.tb_writer.add_scalar(f"f1@t={step}", f1, global_step=self.minibatch_step)
        #     self.tb_writer.add_scalar(f"tp_rate@t={step}", tp_rate, global_step=self.minibatch_step)
        #     self.tb_writer.add_scalar(f"fp_rate@t={step}", fp_rate, global_step=self.minibatch_step)
        #     self.tb_writer.add_scalar(f"tn_rate@t={step}", tn_rate, global_step=self.minibatch_step)
        #     self.tb_writer.add_scalar(f"fn_rate@t={step}", fn_rate, global_step=self.minibatch_step)
        #     self.tb_writer.add_scalar(f"tp@t={step}", true_positives, global_step=self.minibatch_step)
        #     self.tb_writer.add_scalar(f"fp@t={step}", false_positives, global_step=self.minibatch_step)
        #     self.tb_writer.add_scalar(f"tn@t={step}", true_negatives, global_step=self.minibatch_step)
        #     self.tb_writer.add_scalar(f"fn@t={step}", false_negatives, global_step=self.minibatch_step)

        return {
            f'Tactile / accuracy @t={step}': accuracy,
            f'Tactile / precision @t={step}': precision,
            f'Tactile / recall @t={step}': recall,
            f'Tactile / f1 @t={step}': f1,
            
            f'Tactile / true_positive_rate @t={step}': tp_rate,
            f'Tactile / true_negative_rate @t={step}': tn_rate,
            f'Tactile / false_positive_rate @t={step}': fp_rate,
            f'Tactile / false_negative_rate @t={step}': fn_rate,

            f'tp': true_positives,
            f'tn': true_negatives,
            f'fp': false_positives,
            f'fn': false_negatives,
            # f'Tactile / correct @t={step}': correct,
            # f'Tactile / total': total
        }


