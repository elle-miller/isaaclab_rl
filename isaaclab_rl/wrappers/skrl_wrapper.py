# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapper to configure an :class:`ManagerBasedRLEnv` instance to skrl environment.

The following example shows how to wrap an environment for skrl:

.. code-block:: python

    from omni.isaac.lab_tasks.utils.wrappers.skrl import SkrlVecEnvWrapper

    env = SkrlVecEnvWrapper(env, ml_framework="torch")  # or ml_framework="jax"

Or, equivalently, by directly calling the skrl library API as follows:

.. code-block:: python

    from skrl.envs.torch.wrappers import wrap_env  # for PyTorch, or...
    from skrl.envs.jax.wrappers import wrap_env    # for JAX

    env = wrap_env(env, wrapper="isaaclab")

"""

# needed to import for type hinting: Agent | list[Agent]
from __future__ import annotations

from typing import Literal

from isaaclab_rl.wrappers.base_wrapper import Wrapper
from isaaclab_rl.wrappers.isaaclab_wrapper import IsaacLabWrapper


"""
Configuration Parser.
"""


def process_skrl_cfg(cfg: dict, ml_framework: Literal["torch", "jax", "jax-numpy"] = "torch") -> dict:
    """Convert simple YAML types to skrl classes/components.

    Args:
        cfg: A configuration dictionary.
        ml_framework: The ML framework to use for the wrapper. Defaults to "torch".

    Returns:
        A dictionary containing the converted configuration.

    Raises:
        ValueError: If the specified ML framework is not valid.
    """
    _direct_eval = [
        "learning_rate_scheduler",
        "state_preprocessor",
        "value_preprocessor",
        "input_shape",
        "output_shape",
    ]

    def reward_shaper_function(scale):
        def reward_shaper(rewards, timestep, timesteps):
            return rewards * scale

        return reward_shaper

    def update_dict(d):
        # import statements according to the ML framework
        if ml_framework.startswith("torch"):
            # from skrl.resources.preprocessors.torch import RunningStandardScaler  # noqa: F401
            from isaaclab_rl.models.running_standard_scaler import RunningStandardScaler, RunningStandardScalerDict
            from skrl.resources.schedulers.torch import KLAdaptiveLR  # noqa: F401
            from skrl.utils.model_instantiators.torch import Shape  # noqa: F401
        # else:
        #     ValueError(
        #         f"Invalid ML framework for skrl: {ml_framework}. Available options are: 'torch', 'jax' or 'jax-numpy'"
        #     )

        for key, value in d.items():
            if isinstance(value, dict):
                update_dict(value)
            else:
                if key in _direct_eval:
                    d[key] = eval(value)
                elif key.endswith("_kwargs"):
                    d[key] = value if value is not None else {}
                elif key in ["rewards_shaper_scale"]:
                    d["rewards_shaper"] = reward_shaper_function(value)

        return d

    # parse agent configuration and convert to classes
    return update_dict(cfg)


"""
Vectorized environment wrapper.
"""


def SkrlVecEnvWrapper(
    env, ml_framework: Literal["torch", "jax", "jax-numpy"] = "torch", obs_type=None
):
    """Wraps around Isaac Lab environment for skrl.

    This function wraps around the Isaac Lab environment. Since the :class:`ManagerBasedRLEnv` environment
    wrapping functionality is defined within the skrl library itself, this implementation
    is maintained for compatibility with the structure of the extension that contains it.
    Internally it calls the :func:`wrap_env` from the skrl library API.

    Args:
        env: The environment to wrap around.
        ml_framework: The ML framework to use for the wrapper. Defaults to "torch".

    Raises:
        ValueError: When the environment is not an instance of :class:`ManagerBasedRLEnv`.
        ValueError: If the specified ML framework is not valid.

    Reference:
        https://skrl.readthedocs.io/en/latest/api/envs/wrapping.html
    """

    # wrap and return the environment
    return wrap_env(env, wrapper="isaaclab")


def wrap_env(env: Any, wrapper: str = "auto", verbose: bool = True) -> Wrapper:
    """Wrap an environment to use a common interface"""

    def _get_wrapper_name(env, verbose):
        def _in(value, container):
            for item in container:
                if value in item:
                    return True
            return False

        base_classes = [str(base).replace("<class '", "").replace("'>", "") for base in env.__class__.__bases__]
        try:
            base_classes += [
                str(base).replace("<class '", "").replace("'>", "") for base in env.unwrapped.__class__.__bases__
            ]
        except:
            pass
        base_classes = sorted(list(set(base_classes)))

        if _in("omni.isaac.lab.envs.manager_based_env.ManagerBasedEnv", base_classes) or _in(
            "omni.isaac.lab.envs.direct_rl_env.DirectRLEnv", base_classes
        ):
            return "isaaclab"

        return base_classes

    if wrapper == "auto":
        wrapper = _get_wrapper_name(env, verbose)

    if wrapper == "isaaclab" or wrapper == "isaac-orbit":
        return IsaacLabWrapper(env)
    else:
        raise ValueError(f"Unknown wrapper type: {wrapper}")
