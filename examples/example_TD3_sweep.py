import torch

from rsl_rl.algorithms import *
from rsl_rl.env.gym_env import GymEnv
from rsl_rl.runners.runner import Runner
from hyperparams import hyperparams
import wandb
from rsl_rl.runners.callbacks import make_wandb_cb
import numpy as np


ALGORITHM = TD3 # PPO  # DPPO
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# TASK = "BipedalWalker-v3"
TASK = 'HalfCheetah-v4' # "Ant-v4"
PROJECT = "rsl_rl_alg_branch_{}_{}".format(ALGORITHM.__name__, TASK)
TRAIN_ENV_STEPS = 1000000

sweep_config = {
    "method": "random",
    "name": "sweep_{}_{}".format(ALGORITHM.__name__, TASK),
    "metric": {"goal": "maximize", "name": "mean_rewards"},
    "parameters": {
        "algorithm": {"value": ALGORITHM.__name__},
        "task": {"value": TASK},
        "device": {"value": DEVICE},       
        "train_env_steps": {"value": TRAIN_ENV_STEPS},
        ##
        "runner_kwargs": {
            "parameters":{
                "num_steps_per_env": {"values": [16, 24, 50, 200, 1000]}
                }},
        ##
        "env_kwargs": {
            "parameters":{
                "environment_count": {"values": [1, 100, 500]}
            }},
        ##
        "agent_kwargs": {
            "parameters":{
                "actor_activations": {"values": [["relu", "relu", "tanh"], ["relu", "relu", "linear"]]},
                "actor_hidden_dims": {"values": [[256, 256], [64, 64]]},
                "actor_input_normalization": {"values": [False, True]},    
                "action_noise_scale": {"min": 0.01, "max":0.2}, # std of the Gaussian actio noise
                "action_max": {"values": [1, 3.14, 100]},
                "action_min": {"values": [-1, -3.14, -100]},
                "batch_count": {"values":[1, 10, 20]},
                "batch_size": {"values": [64, 100, 200, 500]},
                "critic_activations": {"values": [["relu", "relu", "tanh"], ["relu", "relu", "linear"]]},
                "critic_hidden_dims": {"values": [[256, 256], [64, 64]]},
                "critic_input_normalization": {"values": [False, True]}, 
                "polyak": {"value":0.995},
                # "actor_lr": {"value":1e-3},
                # "critic_lr": {"value":1e-3},
                "actor_lr": {"value":3e-4},
                "critic_lr": {"value":3e-4},
                "noise_clip": {"value":0.5},       # The clipped noise range [-noise_clip, noise_clip]
                "policy_delay": {"value":2},
                "target_noise_scale": {"value":0.2},
                "storage_initial_size": {"value":0}, 
                "storage_size": {"value": 1000000}
            }},
        },
    }
sweep_id = wandb.sweep(sweep=sweep_config, project=PROJECT)

def main():
    print(wandb.config)
    wandb.init()

    train_env_steps = wandb.config.train_env_steps
    runner_kwargs = wandb.config.runner_kwargs
    env_kwargs = wandb.config.env_kwargs
    agent_kwargs = wandb.config.agent_kwargs 

    
    
    learn_steps = None if train_env_steps is None else int(np.ceil(train_env_steps / (env_kwargs["environment_count"] * runner_kwargs["num_steps_per_env"])))
    wandb.config.learn_steps = learn_steps
    print(learn_steps)
    
    env = GymEnv(name=TASK, device=DEVICE, draw=True, **env_kwargs)
 
    wandb_learn_config = dict(
            config=wandb.config,
            group=f"{wandb.config.algorithm}_{wandb.config.task}",
            project=PROJECT,
            tags=[wandb.config.algorithm, wandb.config.task, "train"],
    )


    agent = ALGORITHM(env, benchmark=True, device=DEVICE, **agent_kwargs)
    runner = Runner(env, agent, device=DEVICE, **runner_kwargs)
    runner._learn_cb = [Runner._log]
    runner._learn_cb.append(make_wandb_cb(wandb_learn_config))

    runner.learn(iterations=learn_steps)

wandb.agent(sweep_id, function=main, count=5)

if __name__ == "__main__":
    main()
