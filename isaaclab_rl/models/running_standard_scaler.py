from typing import Optional, Tuple, Union

import gymnasium

import torch
import torch.nn as nn

from skrl import config
from skrl.utils.spaces.torch import compute_space_size
from isaaclab_rl.wrappers.frame_stack import LazyFrames

class RunningStandardScalerDict(nn.Module):
    def __init__(
        self, 
        size: Union[int, Tuple[int], gymnasium.Space],
        epsilon: float = 1e-8,
        clip_threshold: float = 5.0,
        device: Optional[Union[str, torch.device]] = None):
            
        # assert(isinstance(size, dict))
        super(RunningStandardScalerDict, self).__init__()
        self.running_mean_std = nn.ModuleDict({
            k : RunningStandardScaler(v, epsilon, clip_threshold, device) for k,v in size.items()
        })
            
    def forward(
        self, input, train: bool = False, inverse: bool = False, no_grad: bool = True
    ):
        if "policy" in input.keys():
            input = input["policy"]
        res = {k : self.running_mean_std[k](v, train, inverse, no_grad) for k,v in input.items()}
        return res

class RunningStandardScaler(nn.Module):
    def __init__(
        self,
        size: Union[int, Tuple[int], gymnasium.Space],
        epsilon: float = 1e-8,
        clip_threshold: float = 5.0,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """Standardize the input data by removing the mean and scaling by the standard deviation

        The implementation is adapted from the rl_games library
        (https://github.com/Denys88/rl_games/blob/master/rl_games/algos_torch/running_mean_std.py)

        Example::

            >>> running_standard_scaler = RunningStandardScaler(size=2)
            >>> data = torch.rand(3, 2)  # tensor of shape (N, 2)
            >>> running_standard_scaler(data)
            tensor([[0.1954, 0.3356],
                    [0.9719, 0.4163],
                    [0.8540, 0.1982]])

        :param size: Size of the input space
        :type size: int, tuple or list of integers, or gymnasium.Space
        :param epsilon: Small number to avoid division by zero (default: ``1e-8``)
        :type epsilon: float
        :param clip_threshold: Threshold to clip the data (default: ``5.0``)
        :type clip_threshold: float
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or torch.device, optional
        """
        super().__init__()

        self.epsilon = epsilon
        self.clip_threshold = clip_threshold

        self.device = config.torch.parse_device(device)

        print("new standard scaler")

        if type(size) is dict:
            for k,v in size.items():
                print(f"{k} size", v.shape)
                size = v
        else:
            size = compute_space_size(size, occupied_size=True)
        
        print("Registering standard scaler with input size", size)
        self.register_buffer("running_mean", torch.zeros(size, dtype=torch.float64, device=self.device))
        self.register_buffer("running_variance", torch.ones(size, dtype=torch.float64, device=self.device))
        self.register_buffer("current_count", torch.ones((), dtype=torch.float64, device=self.device))

        self.running_mean_min = 0
        self.running_mean_mean = 0
        self.running_mean_max = 0
        self.running_variance_min = 1
        self.running_variance_mean = 1
        self.running_variance_max = 1


    def _parallel_variance(self, input_mean: torch.Tensor, input_var: torch.Tensor, input_count: int) -> None:
        """Update internal variables using the parallel algorithm for computing variance

        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param input_mean: Mean of the input data
        :type input_mean: torch.Tensor
        :param input_var: Variance of the input data
        :type input_var: torch.Tensor
        :param input_count: Batch size of the input data
        :type input_count: int
        """
        delta = input_mean - self.running_mean
        total_count = self.current_count + input_count
        M2 = (
            (self.running_variance * self.current_count)
            + (input_var * input_count)
            + delta**2 * self.current_count * input_count / total_count
        )

        # print("max M2", M2.max().item())
        # print("self.running_variance", self.running_variance.max().item())
        # print("self.current_count", self.current_count.max().item())
        # print(input_var)
        # print("input_var max", input_var.max().item())
        # print("input_var mean", input_var.mean().item())
        # print("input_count", input_count)
        # print()
        # print("delta", delta.max().item())
        # print("1", (self.running_variance * self.current_count).max().item())
        # print("2", (input_var * input_count).max().item())
        # print("3", (delta**2 * self.current_count * input_count / total_count).max().item())

        # update internal variables
        self.running_mean = self.running_mean + delta * input_count / total_count
        self.running_variance = M2 / total_count
        self.current_count = total_count

        self.running_mean_min = self.running_mean.min().item()
        self.running_mean_mean = self.running_mean.mean().item()
        self.running_mean_median = self.running_mean.median().item()

        self.running_mean_max = self.running_mean.max().item()
        self.running_variance_min = self.running_variance.min().item()
        self.running_variance_mean = self.running_variance.mean().item()
        self.running_variance_median = self.running_variance.median().item()
        self.running_variance_max = self.running_variance.max().item()

        # print("updating")
        # print("running var mean", self.running_variance_mean)
        # print("running_variance_min", self.running_variance_min)
        # print("running_variance_median", self.running_variance_median)
        # print("running_variance_max", self.running_variance_max)
        # print("**")


    def _compute(self, x, train: bool = False, inverse: bool = False) -> torch.Tensor:
        """Compute the standardization of the input data

        :param x: Input tensor
        :type x: torch.Tensor
        :param train: Whether to train the standardizer (default: ``False``)
        :type train: bool, optional
        :param inverse: Whether to inverse the standardizer to scale back the data (default: ``False``)
        :type inverse: bool, optional

        :return: Standardized tensor
        :rtype: torch.Tensor
        """

        if isinstance(x, LazyFrames):
            x = x[:]

        if train:
            # x size is [num_samples, obs_size]
            # print(x.shape, x.dim())
            if x.dim() == 3:
                input_mean = torch.mean(x, dim=(0,1)).to(torch.float64)
                input_var = torch.var(x, dim=(0, 1), unbiased=True).to(torch.float64)
                self._parallel_variance(input_mean, input_var, x.shape[0] * x.shape[1])
            elif x.dim() == 2:
                input_mean = torch.mean(x, dim=0).to(torch.float64)
                input_var = torch.var(x, dim=0, unbiased=True).to(torch.float64)
                input_count = x.shape[0]

                # max_index = torch.argmax(input_var)
                # print(max_index)
                # print(x[0, max_index])
                # print(input_mean[max_index])
                # print(input_var[max_index])

                self._parallel_variance(input_mean, input_var, input_count)
            else:
                raise ValueError

        # scale back the data to the original representation
        if inverse:

            # self.check_tensor(x, "x")
            # self.check_tensor(self.running_variance.float(), "var")
            # self.check_tensor(self.running_mean.float(), "mu")
            x = x.to(torch.float64)

            return (
                torch.sqrt(self.running_variance.float())
                * torch.clamp(x, min=-self.clip_threshold, max=self.clip_threshold)
                + self.running_mean.float()
            )
            
        # standardization by centering and scaling
        return torch.clamp(
            (x - self.running_mean.float()) / (torch.sqrt(self.running_variance.float()) + self.epsilon),
            min=-self.clip_threshold,
            max=self.clip_threshold,
        )
    
    def check_tensor(self, tensor, name):
        print(f"Checking {name}:")
        print(f"  - Contains NaN: {torch.isnan(tensor).any()}")
        print(f"  - Contains Inf: {torch.isinf(tensor).any()}")
        print(f"  - Min value: {tensor.min().item()}")
        print(f"  - Max value: {tensor.max().item()}")
        print(f"  - Mean value: {tensor.mean().item()}")


    def forward(
        self, x: torch.Tensor, train: bool = False, inverse: bool = False, no_grad: bool = True
    ) -> torch.Tensor:
        """Forward pass of the standardizer

        Example::

            >>> x = torch.rand(3, 2, device="cuda:0")
            >>> running_standard_scaler(x)
            tensor([[0.6933, 0.1905],
                    [0.3806, 0.3162],
                    [0.1140, 0.0272]], device='cuda:0')

            >>> running_standard_scaler(x, train=True)
            tensor([[ 0.8681, -0.6731],
                    [ 0.0560, -0.3684],
                    [-0.6360, -1.0690]], device='cuda:0')

            >>> running_standard_scaler(x, inverse=True)
            tensor([[0.6260, 0.5468],
                    [0.5056, 0.5987],
                    [0.4029, 0.4795]], device='cuda:0')

        :param x: Input tensor
        :type x: torch.Tensor
        :param train: Whether to train the standardizer (default: ``False``)
        :type train: bool, optional
        :param inverse: Whether to inverse the standardizer to scale back the data (default: ``False``)
        :type inverse: bool, optional
        :param no_grad: Whether to disable the gradient computation (default: ``True``)
        :type no_grad: bool, optional

        :return: Standardized tensor
        :rtype: torch.Tensor
        """
        if no_grad:
            with torch.no_grad():
                return self._compute(x, train, inverse)
        return self._compute(x, train, inverse)
