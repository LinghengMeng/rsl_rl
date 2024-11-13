import torch

from rsl_rl.algorithms import *
from rsl_rl.env.gym_env import GymEnv
from rsl_rl.runners.runner import Runner
from hyperparams import hyperparams


ALGORITHM = PPO  # DPPO
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# TASK = "BipedalWalker-v3"
TASK = 'HalfCheetah-v4' # "Ant-v4"


def main():
    hp = hyperparams[ALGORITHM.__name__][TASK]
    print(hp)
    env = GymEnv(name=TASK, device=DEVICE, draw=True, **hp["env_kwargs"])
    agent = ALGORITHM(env, benchmark=True, device=DEVICE, **hp["agent_kwargs"])
    runner = Runner(env, agent, device=DEVICE, **hp["runner_kwargs"])
    runner._learn_cb = [Runner._log]

    runner.learn(5000)
    
    # env_kwargs = {'environment_count': 1}
    # runner_kwargs = {'num_steps_per_env': 1000}
    # env = GymEnv(name=TASK, device=DEVICE, draw=True, **env_kwargs)
    # agent_kwargs = dict(
    #     actor_activations=["relu", "relu", "tanh"],
    #     actor_hidden_dims=[256, 256],
    #     actor_input_normalization=True,    
    #     action_noise_scale = 0.1, # std of the Gaussian actio noise
    #     action_max = 1,
    #     action_min = -1,
    #     batch_count=1,
    #     batch_size=100,
    #     critic_activations=["relu", "relu", "linear"],
    #     critic_hidden_dims=[256, 256],
    #     critic_input_normalization=True,
    #     polyak = 0.995,
    #     actor_lr = 1e-3,
    #     critic_lr = 1e-3,
    #     noise_clip = 0.5,       # The clipped noise range [-noise_clip, noise_clip]
    #     policy_delay = 2,
    #     target_noise_scale = 0.2,
    #     storage_initial_size = 0, 
    #     storage_size = 1000000
    # )
    # agent = ALGORITHM(env, benchmark=True, device=DEVICE, **agent_kwargs)

    # runner = Runner(env, agent, device=DEVICE, **runner_kwargs)
    # runner._learn_cb = [Runner._log]

    # runner.learn(5000000)


if __name__ == "__main__":
    main()
