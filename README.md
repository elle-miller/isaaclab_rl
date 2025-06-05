# isaaclab_rl

A library for training robotic agents in Isaac Lab with PPO, with in-built hyperparameter optimisation, and extensive logging and plotting options. This is a twin library with [isaaclab_rl_project](https://github.com/elle-miller/isaaclab_rl_project), which contains an example environment and blank template environment.

![isaaclab_rl](https://github.com/user-attachments/assets/72036a2f-41ab-4317-ad30-8a165afa83a5)

**Features**
- Dictionary observations (makes life easy if you have different observation types you want to swap in and out)
- Wrappers for observation stacking (important for partially observable envs!)
- Split environments for training and evaluation (will get more accurate measure of agent learning)
- All RL related code is simplified into 4 files (you can easily figure out what's going on and edit)

## Installation

1. Install Isaac Lab via pip with [these instructions](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/isaaclab_pip_installation.html)

2. Install `isaaclab_rl` as a local editable package.

```
git clone git@github.com:elle-miller/isaaclab_rl.git
cd isaaclab_rl
pip install -e .
```
You should now see it with `pip show isaaclab_rl`.

3. Setup your own project!

Now follow the README instructions here [isaaclab_rl_project](https://github.com/elle-miller/isaaclab_rl_project).


## Motivation for this library

Are you a researcher wanting to get into using RL with Isaac Lab as painlessly as possible? This library is for you!

There are many libraries with various RL implementations out there. However, many of these libraries do not provide support for doing robust RL research, such as reporting mean evaluation returns, correct number of timesteps, or providing integrated hyperparameter optimisation. These are well established norms in the RL research community, but are not yet consistently present in RL+robotics research. This was the library I made for my own PhD research, and am open-sourcing it to avoid others having to repeat re-implement all these components :)

## How it works

`isaaclab_rl` contains all the core components to run RL in Isaac Lab, that I will continuously add to and improve. Your own project `isaaclab_rl_project` runs on these modules, but is completely separated so you can do what you like, and optionally pull changes from `isaaclab_rl`. 

`isaaclab_rl` provides 4 core functionalities:

1. **algorithms**: anything RL related is here
2. **models**: base models used by RL e.g. MLP, CNN, running standard scaler 
3. **tools**: scripts to produce those nice RL paper plots, and extra stuff like latent trajectory visualisation.
4. **wrappers**: wrappers for observation stacking and isaaclab

## Credits
The PPO implementation is a streamlined version of the one provided by [SKRL](https://github.com/Toni-SM/skrl), full credits to toni-sm for this. The reason I am providing a local version instead of importing from SKRL is because there are major breaking changes, e.g. all irrelevant functions are deleted, logic is different to be able to reporting mean evaluation returns.  


## 📚 Citation
If this code has been helpful for your research, please cite:

```
@misc{miller2025_isaaclab_rl,
  author       = {Elle Miller},
  title        = {isaaclab_rl},
  year         = {2025},
  howpublished = {\url{https://github.com/elle-miller/isaaclab_rl}},
  note         = {GitHub repository}
}
```
 
