import torch
import torch.nn as nn
import torch.nn.functional as F

from isaaclab_rl.models.cnn import ImageEncoder
from isaaclab_rl.models.mlp import MLP
from isaaclab_rl.models.running_standard_scaler import RunningStandardScalerDict


methods = ["early", "intermediate"]


class Encoder(nn.Module):
    """encoder"""

    def __init__(self, observation_space, action_space, env_cfg, config_dict, device):
        super().__init__()

        print(config_dict)

        self.method = config_dict["encoder"]["method"]
        assert self.method in methods
        self.device = device

        self.observation_space = observation_space
        self.action_space = action_space

        self.num_inputs = 0

        self.hiddens = config_dict["encoder"]["hiddens"]
        self.activations = config_dict["encoder"]["activations"]

        # standard scaler
        if config_dict["encoder"]["state_preprocessor"] is not None:
            self.state_preprocessor = RunningStandardScalerDict(size=observation_space, device=device)
        else:
            self.state_preprocessor = None

        # configure relevant preprocessing
        for k, v in observation_space.items():

            if k == "pixels":
                # always use cnn for pixels
                self.pixel_obs_dim = v.shape
                self.img_dim = config_dict["pixels"]["img_dim"]
                latent_img_dim = config_dict["pixels"]["latent_img_dim"]
                self.cnn = ImageEncoder(self.pixel_obs_dim, latent_dim=latent_img_dim, num_layers=2).to(device)
                self.num_inputs += latent_img_dim
                # print("***********CNN*************")
                # print(self.cnn)

            # TODO: figure out prop + gt case
            elif k == "prop":
                num_prop_inputs = v.shape[0]
                if self.method == "early":
                    self.num_inputs += num_prop_inputs
                # make mlp for prop input
                elif self.method == "intermediate":
                    raise NotImplementedError

            elif k == "gt":
                num_gt_inputs = v.shape[0]
                if self.method == "early":
                    self.num_inputs += num_gt_inputs
                # make mlp for prop input
                elif self.method == "intermediate":
                    raise NotImplementedError

            elif k == "tactile":
                num_tactile_inputs = v.shape[0]
                if self.method == "early":
                    self.num_inputs += num_tactile_inputs
                # make mlp for prop input
                elif self.method == "intermediate":
                    raise NotImplementedError
        
        # lstm layer 
        # self.lstm = torch.nn.LSTM(self.num_inputs, self.num_inputs, num_layers=1, bias=True, batch_first=False)

        if self.hiddens == []:
            self.net = nn.Sequential(nn.Identity())
            self.num_outputs = self.num_inputs

        else:
            layernorm = config_dict["encoder"]["layernorm"]
            self.num_outputs = self.hiddens[-1]
            self.net = MLP(
                self.num_inputs,
                self.hiddens,
                self.activations,
                layernorm=layernorm
            ).to(device)

            # self.net = ResidualEncoder(
            #     input_dim=self.num_inputs,
            #     latent_dim=self.num_outputs,
            # ).to(device)

        # self.net.add_module("ln", torch.nn.LayerNorm(self.num_outputs).to(device))

        # self.net.add_module("dropout", torch.nn.Dropout(p=0.3).to(device))

        # print("*********Encoder*************")
        # print(self.net)

    def concatenate_obs(self, obs_dict):
        # separate out components of obs dict
        # for early , concat raw inputs with image inputs
        if self.method == "early":
            raw_inputs = self.get_raw_inputs(obs_dict)
            latent_inputs = self.get_latent_inputs(obs_dict)
            concat_obs = torch.cat((raw_inputs, latent_inputs), dim=-1)

        # for intermediate , pass raw inputs through mlps
        else:
            raise NotImplementedError
        
        return concat_obs

    def forward(self, obs_dict, detach=False, train=False):
        """
        Take in an obs dict, and return z
        
        """
        # sometimes need to detach, e.g. for linear probing
        if detach:
            obs_dict = {key: value.detach() for key, value in obs_dict.items()}

        if "policy" in obs_dict.keys():
            obs_dict = obs_dict["policy"]

        # scale inputs
        if self.state_preprocessor is not None:
            obs_dict = self.state_preprocessor(obs_dict, train)

        concat_obs = self.concatenate_obs(obs_dict)

        # h, _ = self.lstm(concat_obs)
        z = self.net(concat_obs)

        if z.dim() == 1:
            z = z.unsqueeze(-1)  # Adds a trailing dimension to ensure (num_envs, obs) when only one observation

        return z

    def get_raw_inputs(self, obs_dict):
        """
        Retrieve prop or gt if exists
        """
        raw_inputs = torch.tensor([]).to(self.device)
        # LOOP THROUGH DICT IN ALPHABETICAL ORDER!!!!!!!!!!!!!!!!!! fml
        for k in sorted(obs_dict.keys()):
            if k == "prop" or k == "gt" or k == "tactile":
                raw_inputs = torch.cat((raw_inputs, obs_dict[k][:]), dim=-1)
        return raw_inputs

    def get_latent_inputs(self, obs_dict):

        latent_inputs = torch.tensor([]).to(self.device)
        for k in sorted(obs_dict.keys()):
            if k == "pixels" or k == "depth":
                # assuming a shared cnn at the moment
                z = self.cnn(obs_dict[k][:])
                latent_inputs = torch.cat((latent_inputs, z), dim=-1)
        return latent_inputs

