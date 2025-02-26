from datetime import datetime
import git
import os
import numpy as np
import pathlib
import random
import torch


def environment_dimensions(env):
    obs = env.get_observations()

    if isinstance(obs, tuple):
        obs, env_info = obs
    else:
        env_info = {}

    dims = {}

    dims["observations"] = obs.shape[1]

    if "observations" in env_info and "critic" in env_info["observations"]:
        dims["actor_observations"] = dims["observations"]
        dims["critic_observations"] = env_info["observations"]["critic"].shape[1]
    else:
        dims["actor_observations"] = dims["observations"]
        dims["critic_observations"] = dims["observations"]

    dims["actions"] = env.num_actions
    # Get the number of reward terms
    dims["reward_term_num"] = env.num_reward_terms

    return dims


def split_and_pad_trajectories(tensor, dones):
    """Splits trajectories at done indices. Then concatenates them and padds with zeros up to the length og the longest trajectory.
    Returns masks corresponding to valid parts of the trajectories
    Example:
        Input: [ [a1, a2, a3, a4 | a5, a6],
                 [b1, b2 | b3, b4, b5 | b6]
                ]

        Output:[ [a1, a2, a3, a4], | [  [True, True, True, True],
                 [a5, a6, 0, 0],   |    [True, True, False, False],
                 [b1, b2, 0, 0],   |    [True, True, False, False],
                 [b3, b4, b5, 0],  |    [True, True, True, False],
                 [b6, 0, 0, 0]     |    [True, False, False, False],
                ]                  | ]

    Assumes that the inputy has the following dimension order: [time, number of envs, aditional dimensions]
    """
    dones = dones.clone()
    dones[-1] = 1
    # Permute the buffers to have order (num_envs, num_transitions_per_env, ...), for correct reshaping
    flat_dones = dones.transpose(1, 0).reshape(-1, 1)

    # Get length of trajectory by counting the number of successive not done elements
    done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero()[:, 0]))
    trajectory_lengths = done_indices[1:] - done_indices[:-1]
    trajectory_lengths_list = trajectory_lengths.tolist()
    # Extract the individual trajectories
    trajectories = torch.split(tensor.transpose(1, 0).flatten(0, 1), trajectory_lengths_list)
    padded_trajectories = torch.nn.utils.rnn.pad_sequence(trajectories)

    trajectory_masks = trajectory_lengths > torch.arange(0, tensor.shape[0], device=tensor.device).unsqueeze(1)
    return padded_trajectories, trajectory_masks


def unpad_trajectories(trajectories, masks):
    """Does the inverse operation of  split_and_pad_trajectories()"""
    # Need to transpose before and after the masking to have proper reshaping
    return (
        trajectories.transpose(1, 0)[masks.transpose(1, 0)]
        .view(-1, trajectories.shape[0], trajectories.shape[-1])
        .transpose(1, 0)
    )


def store_code_state(logdir, repositories):
    git_log_dir = os.path.join(logdir, "git")
    os.makedirs(git_log_dir, exist_ok=True)
    file_paths = []
    for repository_file_path in repositories:
        repo = git.Repo(repository_file_path, search_parent_directories=True)
        repo_name = pathlib.Path(repo.working_dir).name
        t = repo.head.commit.tree
        diff_file_name = os.path.join(git_log_dir, f"{repo_name}.diff")
        # check if the diff file already exists
        if os.path.isfile(diff_file_name):
            continue
        # write the diff file
        print(f"Storing git diff for '{repo_name}' in: {diff_file_name}")
        content = f"--- git status ---\n{repo.git.status()} \n\n\n--- git diff ---\n{repo.git.diff(t)}"
        with open(diff_file_name, "x", encoding="utf-8") as f:
            f.write(content)
        # add the file path to the list of files to be uploaded
        file_paths.append(diff_file_name)
    return file_paths


def seed(s=None):
    seed = int(datetime.now().timestamp() * 1e6) % 2**32 if s is None else s

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def squeeze_preserve_batch(tensor):
    """Squeezes a tensor, but preserves the batch dimension"""
    single_batch = tensor.shape[0] == 1

    squeezed_tensor = tensor.squeeze()

    if single_batch:
        squeezed_tensor = squeezed_tensor.unsqueeze(0)

    return squeezed_tensor
