import numpy as np
import torch
from typing import Tuple, Optional
from isaaclab_rl.wrappers.frame_stack import LazyFrames


class PrioritizedMemory:
    def __init__(self,
                 env,
                 encoder,
                 value,
                 value_preprocessor,
                 memory_type: str,
                 memory_size: int,
                 seq_length: int = 1, 
                 metric: str = "return",
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 beta_annealing_steps: int = 100000,
                 epsilon: float = 1e-5,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Prioritized experience replay buffer for storing state-action transitions.
        
        Args:
            memory_size: Maximum number of transitions to store
            seq_length: Length of each transition sequence
            obs_size: Dimensionality of observations
            action_size: Dimensionality of actions
            alpha: Prioritization exponent (0 = uniform sampling, 1 = fully prioritized)
            beta: Importance sampling correction exponent
            beta_annealing_steps: Steps over which to anneal beta to 1.0
            epsilon: Small constant to add to priorities to ensure non-zero sampling probability
            device: Device to store tensors on
        """
        self.encoder = encoder
        self.value = value 
        self.value_preprocessor = value_preprocessor 

        self.memory_type = memory_type
        assert memory_type == "reduced_vanilla" or memory_type == "prioritised" or memory_type == "n_vanilla"

        self.memory_size = memory_size
        self.seq_length = seq_length
        self.metric = metric

        self.alpha = alpha
        self.beta = beta
        self.beta_annealing_steps = beta_annealing_steps
        self.beta_increment = (1.0 - beta) / beta_annealing_steps
        self.epsilon = epsilon
        self.device = device

        self.env = env
        self.gamma = 0.99
        # Initialize memory buffers
        print(self.env.action_space.shape[0])
        print((memory_size, seq_length, self.env.action_space.shape[0]))

        # to do: try float16 ????
        self.dtype = torch.float32

        if self.memory_type == "reduced_vanilla" or self.memory_type == "n_vanilla":
            self.init_importance = 0
        
        # set 0 or messes up measurements
        elif self.memory_type == "prioritised":
            self.init_importance = 0
        
    def reset(self):
        # print("reseting aux memory")
        self.actions = torch.zeros((self.memory_size, self.seq_length, self.env.action_space.shape[0]), device=self.device, dtype=self.dtype)
        self.importance = torch.ones(self.memory_size, device=self.device, dtype=self.dtype) * self.init_importance
        self.returns = torch.zeros(self.memory_size, device=self.device, dtype=self.dtype)
        self.td_errors = torch.zeros(self.memory_size, device=self.device, dtype=self.dtype)

        observation_names = []
        self.states = {}
        # outer loop of observation space (policy, aux)
        for type_k in sorted(self.env.observation_space.keys()):
            for k, v in self.env.observation_space[type_k].items():
                # create next states for the forward_dynamics
                # print(f"AuxiliaryTask: {k}: {type_k} tensor size {v.shape}")
                # forward dynamics use sequence length
                self.states[k] = torch.zeros((self.memory_size, self.seq_length, *v.shape), device=self.device, dtype=self.dtype)
                # recon we don't
                # self.states[k] = torch.zeros((self.memory_size, *v.shape), device=self.device, dtype=self.dtype)
                observation_names.append(k)
        
        # Initialize tracking variables
        self.memory_index = 0
        self.filled = False
        self.total_samples = 0

    def __len__(self) -> int:
        """Compute and return the current (valid) size of the memory

        The valid size is calculated as the ``memory_size * num_envs`` if the memory is full (filled).
        Otherwise, the ``memory_index * num_envs + env_index`` is returned

        :return: Valid size
        :rtype: int
        """
        return self.memory_size if self.filled else self.memory_index
        
    def add_samples(self, incoming_states, incoming_actions: torch.Tensor,
            incoming_importances: Optional[float] = None) -> int:
        """
        Add a new transition to the buffer.
        
        Args:
            state: State tensor [seq_length, obs_size]
            action: Action tensor [seq_length, action_size]
            next_state: Next state tensor [seq_length, obs_size]
            importance: Importance metric (if None, uses max priority)
            
        Returns:
            Index where the transition was stored
        """
        
        # mean_importance is initially set to 0.5
        self.mean_importance = self.get_mean_importance()

        # add everything
        if self.memory_type == "reduced_vanilla" or self.memory_type == "n_vanilla":
            self.add_all_samples(incoming_states, incoming_actions, incoming_importances)
        # add based on performance
        elif self.memory_type == "prioritised":
            self.add_all_samples(incoming_states, incoming_actions, incoming_importances)

            # if not self.filled:
            #     self.add_all_samples(incoming_states, incoming_actions, incoming_importances)
            # else:
            #     self.add_priority_samples(incoming_states, incoming_actions, incoming_importances)
        else:
            raise ValueError
        
        

    def add_all_samples(self, incoming_states, incoming_actions, incoming_returns):
        # note this method assumes we will always have "prop" obs
        obs = incoming_states["prop"]

        # we do NOT save as LazyFrames [samples are already filtered by this point]
        if isinstance(obs, LazyFrames):
            raise TypeError("sort out LazyFrames before this please")
            
        if len(obs.shape) == 3:
            num_incoming_samples, seq_length, size = obs.shape
        elif len(obs.shape) == 2:
            num_incoming_samples, size = obs.shape
        else:
            raise ValueError

        self.total_samples += num_incoming_samples

        # if we are not filled, add relevant samples sequentially to the memory index
        num_samples = min(num_incoming_samples, self.memory_size - self.memory_index)
        overflow_samples = num_incoming_samples - num_samples
        # print(f"adding {num_samples} seq to a memory index {self.memory_index}. overflow_samples {overflow_samples}") #, remaining_samples)

        # Store transition
        for k, v in incoming_states.items():
            self.states[k][self.memory_index : self.memory_index + num_samples] = v[:][:num_samples]

            if overflow_samples > 0:
                overflow_data = v[:][num_samples:num_samples+overflow_samples]
                self.states[k][:overflow_samples] = overflow_data

        self.actions[self.memory_index : self.memory_index + num_samples] = incoming_actions[:num_samples]
        self.returns[self.memory_index : self.memory_index + num_samples] = incoming_returns[:num_samples]

        # don't even bother with the overflow
        if overflow_samples > 0:
            self.actions[:overflow_samples] = incoming_actions[num_samples:]
            self.returns[:overflow_samples] = incoming_returns[num_samples:]
            self.memory_index = overflow_samples
            self.filled = True
            # self.memory_index = self.memory_size
            # print("Aux memory is now filled!")
        else:
            self.memory_index += num_samples

    
    # def add_priority_samples(self, incoming_states, incoming_actions, incoming_returns):

    #     # add samples higher than the mean
    #     above_mean_importance_mask = incoming_returns >= self.mean_importance

    #     # Find which samples are important enough to add
    #     num_to_add = above_mean_importance_mask.sum().item()
    #     self.total_samples += num_to_add
        
    #     if num_to_add == 0:
    #         print("nothing to add")
    #         return   # Nothing to add
        
    #     # Get indices of least important samples to replace
    #     replace_indices = self.get_least_important_indices(num_to_add)

    #     # Find which samples to add (those above mean importance)
    #     important_sample_indices = torch.where(above_mean_importance_mask)[0]

    #     # Batch update all states at once
    #     for k, v in incoming_states.items():
    #         # Use advanced indexing to replace all values at once
    #         self.states[k][replace_indices] = v[important_sample_indices].to(self.states[k].dtype)

    #     # Batch update actions and importance
    #     self.actions[replace_indices] = incoming_actions[important_sample_indices].to(self.actions.dtype)
    #     self.incoming_returns[replace_indices] = incoming_returns[important_sample_indices].to(self.importance.dtype)
            
    #     # mean importance
    #     self.mean_importance = self.get_mean_importance()

    # def get_least_important_indices(self, n):
    #     """Get indices of the n least important samples in the buffer."""

    #     # Get indices of n smallest values
    #     _, indices = torch.topk(self.importance, n, largest=False)
    #     return indices
           

    def sample_all(self, mini_batches: int, update_beta: bool = True) -> Tuple[torch.Tensor, ...]:
        """
        Sample a batch of transitions based on their importance.
        
        Args:
            batch_size: Number of transitions to sample
            update_beta: Whether to update beta parameter for importance sampling
            
        Returns:
            Tuple of (states, actions, next_states, indices, weights)
        """
        # Determine actual memory size (full size or counter)
        # actual_size = self.memory_size if self.filled else self.memory_index
        
        # # Calculate sampling probabilities based on importance
        # priorities = self.importance[:actual_size] ** self.alpha
        # probs = priorities / priorities.sum()
        
        # # Sample indices based on probabilities
        # indices = torch.multinomial(probs, batch_size, replacement=True)
        
        # # Calculate importance sampling weights
        # weights = (actual_size * probs[indices]) ** (-self.beta)
        # weights = weights / weights.max()  # Normalize to stabilize training

        # self.update_importances()

        mem_size = self.memory_size if self.filled else self.memory_index
        batch_size = mem_size // mini_batches

        if self.memory_type == "prioritised":

            temperature = 0.5

            # Handle negative values by adding an offset
            if not self.filled:
                importance_vals = self.importance[:mem_size].cpu().numpy()
            else:
                importance_vals = self.importance.cpu().numpy()

            # Handle negative values
            offset = max(0.0, abs(np.min(importance_vals)) + 1e-6)
            shifted_vals = importance_vals + offset
            
            # Apply exponential weighting with temperature
            weights = np.exp(shifted_vals / temperature)

            # print("max, mean, min weights:", np.max(weights), np.mean(weights), np.min(weights))

            # Normalize to get probabilities
            probs = weights / weights.sum()

            # print("max, mean, min prob:", np.max(probs), np.mean(probs), np.min(probs))
            
            # Sample based on probabilities
            indexes = np.random.choice(
                mem_size,
                size=mem_size,
                replace=True,
                p=probs
            )

        else:
            # sample every single guy in memory
            indexes = np.arange(mem_size)
            np.random.shuffle(indexes)
               
        indexes = indexes.tolist()

        # Split into minibatches
        batches = [indexes[i:i+batch_size] for i in range(0, len(indexes) - batch_size + 1, batch_size)]

        minibatches = []
        for batch in batches:
            minibatch = []
            minibatch_obs_dict = {}

            for k in self.env.observation_space["policy"].keys():
                minibatch_obs_dict[k] = self.states[k][batch]

            minibatch_actions = self.actions[batch]
            minibatch_importance = self.importance[batch]

            minibatch = [minibatch_obs_dict, minibatch_actions, minibatch_importance]
    
            minibatches.append(minibatch)

        return minibatches

    
    def compute_td_errors(self) -> float:
        """Compute TD errors in memory-efficient batches"""
        
        # Initialize an empty tensor to store the results
        td_errors = torch.zeros_like(self.returns)
        
        # Process in batches to avoid OOM
        batch_size = 32768  # Adjust based on your memory constraints
        num_sequences = self.memory_size if self.filled else self.memory_index
        
        for batch_start in range(0, num_sequences, batch_size):
            batch_end = min(batch_start + batch_size, num_sequences)
            batch_indices = slice(batch_start, batch_end)
            
            # Get returns for this batch
            batch_returns = self.returns[batch_indices]
            
            # Get first states in sequence for this batch
            # Process with no_grad to save memory if just for prioritization
            with torch.no_grad():
                # these are dictionaries!
                batch_first_obs = self.get_obs_dict_list(self.states)[0]
                batch_last_obs = self.get_obs_dict_list(self.states)[-1]
                        
                # Encode and compute values
                first_z = self.encoder(batch_first_obs)[batch_indices]
                first_v = self.value.compute_value(first_z)
                first_v = self.value_preprocessor(first_v, inverse=True)
                
                last_z = self.encoder(batch_last_obs)[batch_indices]
                last_v = self.value.compute_value(last_z)
                last_v = self.value_preprocessor(last_v, inverse=True)

            # Compute TD errors for this batch
            batch_td_errors = batch_returns - first_v.squeeze() + self.gamma * last_v.squeeze()
            
            # If needed, process middle states (more memory intensive)
            if self.seq_length > 2:
                for i in range(1, self.seq_length-1):
                    batch_middle_obs = self.get_obs_dict_list(self.states)[i]
                    with torch.no_grad():
                        middle_z = self.encoder(batch_middle_obs)[batch_indices]
                        middle_v = self.value.compute_value(middle_z)
                        middle_v = self.value_preprocessor(middle_v, inverse=True)

                    batch_td_errors -= -middle_v.squeeze() * (1-self.gamma)
            
            # Store the results
            td_errors[batch_indices] = batch_td_errors
        
        self.td_errors = td_errors
        return td_errors


    def get_buffer_size(self) -> int:
        """Return the current number of transitions stored."""
        return self.memory_size if self.filled else self.memory_index

    def get_mean_importance(self) -> float:
        # Compute current mean importance
        self.mean_importance = self.importance.mean()
        return self.mean_importance

    def update_importances(self):
        # NORMALISE 0 TO 1
        if self.metric == "return":
            raw_importance = self.returns.clone()
        elif self.metric == "td":
            self.compute_td_errors()
            raw_importance = self.td_errors.clone()
        else:
            raise ValueError

        min_val = raw_importance.min()
        max_val = raw_importance.max()
        self.importance = (raw_importance - min_val) / (max_val - min_val)
    
    def get_mean_return(self) -> float:
        # Compute current mean importance
        self.mean_return = self.returns.mean()
        return self.mean_return
    
    def get_mean_td_error(self) -> float:
        # Compute current mean
        self.mean_td_error = self.td_errors.mean()
        return self.mean_td_error
        

    def get_obs_dict_list(self, obs_sequence):
        # get individual obs_dict for each state in order to get z
        obs_dict_list = [{} for _ in range(self.seq_length)]
        for k in sorted(obs_sequence.keys()):
            # shape (batch_size, seq_length, input_size)
            obs = obs_sequence[k] 
            for i in range(self.seq_length):
                # saving as (batch_size, input_size)
                obs_dict_list[i][k] = obs[:,i,:]
        return obs_dict_list
    
