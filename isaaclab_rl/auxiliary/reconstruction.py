import torch
import torch.nn as nn
import torch.nn.functional as F

from isaaclab_rl.auxiliary.task import AuxiliaryTask
from isaaclab_rl.auxiliary.physics_memory import PrioritizedMemory

from isaaclab_rl.wrappers.frame_stack import LazyFrames



class nDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(nDecoder, self).__init__()

        # output_dim = n*output_dim

        if output_dim > 64:

            self.net = nn.Sequential(
            
                nn.Linear(latent_dim, 512),
                nn.LayerNorm(512),
                nn.ELU(alpha=1.0),
                
                nn.Linear(512, 512),
                nn.LayerNorm(512),
                nn.ELU(alpha=1.0),
                
                nn.Linear(512, output_dim),
            )
            
        else:
            self.net = nn.Sequential(
                
                nn.Linear(latent_dim, 128),
                nn.LayerNorm(128),
                nn.ELU(alpha=1.0),
                
                nn.Linear(128, 64),
                nn.LayerNorm(64),
                nn.ELU(alpha=1.0),
                
                nn.Linear(64, output_dim),
            )

    def forward(self, x):
        output = self.net(x)
        return output
        
         

class Reconstruction(AuxiliaryTask):
    """
    Reconstruct pixel inputs.
    Note only current works with tasks where only observation is pixels.
    """

    def __init__(self, aux_task_cfg, rl_rollout, rl_memory, encoder, value, value_preprocessor, env, env_cfg, writer):
        super().__init__(aux_task_cfg, rl_rollout, rl_memory, encoder, value, value_preprocessor, env, env_cfg, writer)

        # reconstruct everything
        num_inputs = self.encoder.num_outputs
        num_outputs = self.encoder.num_inputs

        self.num_prop_obs = int(self.env.num_prop_observations / self.env.obs_stack)
        self.num_tactile_obs = int(self.env.num_tactile_observations / self.env.obs_stack)
        self.binary = self.env.binary_tactile
        print("****Reconstruction network*****")
        print("tactile size", self.num_tactile_obs)
        print("prop size", self.num_prop_obs)
        print("reconstructing last", self.last_n_obs)

        if self.tactile_only:
            output_dim = self.num_tactile_obs * self.last_n_obs

        else:
            output_dim = (self.num_prop_obs + self.num_tactile_obs) * self.last_n_obs
        
        self.decoder = nDecoder(latent_dim=num_inputs, output_dim=output_dim).to(self.device)
        print(self.decoder)

        self.counter = 0

        super()._post_init()

    def set_optimisable_networks(self):
        return [self.encoder, self.decoder]
        # return [self.decoder]

    def create_memory(self):
        return PrioritizedMemory(self.env, self.memory_size) 
        # return self.create_sequential_memory(size=self.memory_size)

    def sample_minibatches(self, minibatches):
        batch_list = []
        sampled_batches = self.memory.sample_all(mini_batches=minibatches)
        batch_list.append(sampled_batches)
        return batch_list
    
    # Check for NaN/Inf in your inputs
    def check_tensor(self, tensor, name):
        print(f"Checking {name}:")
        print(f"  - Contains NaN: {torch.isnan(tensor).any()}")
        print(f"  - Contains Inf: {torch.isinf(tensor).any()}")
        print(f"  - Min value: {tensor.min().item()}")
        print(f"  - Max value: {tensor.max().item()}")
        print(f"  - Mean value: {tensor.mean().item()}")

    def compute_loss(self, minibatch):
        """
        Compute loss on minibatch
        
        """
        # in this case, states contains sequences of transitions
        states, actions = minibatch
        states, next_states = self.separate_memory_tensors(states)

        tactile_coefficient = 1

        # ground truth
        obs_dict = states["policy"]

        # print(obs_dict["tactile"].shape)
        # most recent tactile obs are at the end
        n_tactile = self.last_n_obs * self.num_tactile_obs
        current_tactile_obs = obs_dict["tactile"][:,-n_tactile:]
        tactile_true = current_tactile_obs
        n_prop = self.last_n_obs * self.num_prop_obs
        current_prop_obs = obs_dict["prop"][:,-n_prop:]
        prop_true = current_prop_obs
        
        # predictions
        z = self.encoder(obs_dict)
        x_hat = self.decoder(z)

        if self.tactile_only:
            tactile_pred = x_hat
        else:
            # split
            tactile_pred = x_hat[:, n_prop:]
            prop_pred = x_hat[:, :n_prop]

        if self.binary:

            # sigmoid the tactile pred
            tactile_pred = torch.sigmoid(tactile_pred)

            # Use BCE loss for binary tactile signals 
            # If 1s are ~10x less common than 0s across all positions
            pos_weight = torch.ones(n_tactile).to(self.device) * 10
            tactile_loss = F.binary_cross_entropy_with_logits(
                tactile_pred, tactile_true, 
                pos_weight=pos_weight  # Applies to all positions equally
            )

        else:
            # Use MSE for continuous proprioception signals
            tactile_loss = F.mse_loss(tactile_pred, tactile_true)

        if not self.tactile_only:
            prop_loss = F.mse_loss(prop_pred, prop_true)
        else:
            prop_loss = 0

        loss = tactile_loss + prop_loss

        self.counter += 1

        info = {
            "Loss / Recon_tactile_loss": tactile_loss.item()
            }
        
        if self.binary:
            more_info = self.evaluate_binary_predictions(tactile_pred, tactile_true)
            info.update(more_info)
        if not self.tactile_only:
            more_info = {"Loss / Recon_prop_loss": prop_loss.item()}
            info.update(more_info)


        return loss, info
    


    def add_samples(self, states):
        """
        Add samples to dedicated aux memory
        Re-implement this if you don't need all of these tensors

        Samples must come in as Lazy Frames

        But be saved in their expanded form
        
        """
        if not isinstance(states["policy"]["prop"], LazyFrames):
            raise TypeError("should be LazyFrames")
        
        states = states["policy"]

        # don't need to worry about alive/dead for reconstruction
        for obs_k in self.env.observation_space["policy"].keys():
            if isinstance(states[obs_k], LazyFrames):
                states[obs_k] = states[obs_k][:]

        # only add important samples
        if self.memory_type == "reduced_vanilla":
            importance = self.env.compute_reconstruction_transition_importance(states)

        # else, everything is important
        else:
            importance = torch.zeros((states["prop"].shape[0],), device=self.device, dtype=torch.float16)

        self.memory.add_samples(
            states,
            None,
            importance
        )

        # sampled_states = sampled_states["policy"]
        # for k, v in sampled_states.items():
        #     z = self.encoder(sampled_states) 

        #     if k == "pixels":
        #         prediction = self.pixel_decoder(z)
        #         target = sampled_states["pixels"] / 255.0
        #         loss = self.criterion(
        #             prediction,
        #             target.permute((0, 3, 1, 2)) if not self.encoder.cnn.channels_first else target,
        #         )

        #         self.save_reconstructed_images(prediction, wandb_session)
        #         self.pixel_decoder_optimiser.zero_grad()
        #         loss.backward()
        #         self.pixel_decoder_optimiser.step()

        #     # elif k == "prop":
        #     #     # question of normalisation... 
        #     #     prop_pred = self.prop_decoder(z)

        #     #     loss = self.criterion(prop_pred, sampled_states["prop"])

        #     #     # if self.counter % 100 == 0:
        #     #     #     print(self.counter, "Example prediction: ", loss.item())
        #     #     #     print(prop_pred[0])
        #     #     #     print(sampled_states["prop"][0])

        #     #     #     print("****")
        #     #     #     print(self.prop_decoder[0].weight)  # Prints the weight matrix of the Linear layer
        #     #     #     print(self.prop_decoder[0].bias)    # Prints the bias vector of the Linear layer

        #     #     self.prop_decoder_optimiser.zero_grad()
        #     #     loss.backward()
        #     #     self.prop_decoder_optimiser.step()
        
        # self.counter += 1

        
    
    # def save_reconstructed_images(self, recon, wandb_session):
        
    #     if (self.counter % 100) == 0:
    #         random_env_id = 10  # torch.randint(0, int(sampled_states["pixels"].size()[0]), (1,))
    #         save_image_to_wandb(
    #             wandb_session,
    #             recon[random_env_id],
    #             caption="reconstruction",
    #             step=self.counter,
    #         )

