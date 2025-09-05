import os, pickle, glob
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

from tqdm import tqdm

from model.normalize import normalize
from model.normalize import denormalize
from matplotlib import pyplot as plt





def visualize_sample(obs, action, num_points=51):
    """
    Visualize the observation and action from a single sample.
    
    Args:
        obs: tensor of shape (1, 1, 204) - flattened initial and target shapes
        action: tensor of shape (1, 16, 4) - sequence of actions [idx, x, y, theta]
        num_points: number of particles (51)
    """
    # Remove batch dimension and observation horizon dimension
    obs = obs.squeeze()  # Shape: (204,)
    action = action.squeeze()  # Shape: (16, 4)
    
    # Split observation into initial and target shapes
    coords_per_shape = num_points * 2  # 51 points * 2 coordinates (x, y)
    
    initial_coords = obs[:coords_per_shape].reshape(num_points, 2)  # (51, 2)
    target_coords = obs[coords_per_shape:].reshape(num_points, 2)   # (51, 2)
    
    # Extract action information
    action_idx = int(action[0, 0])  # Index of the particle being manipulated
    action_displacements = action[:, 1:3]  # (16, 2) - x, y displacements
    
    # Get starting position of the manipulated particle
    start_pos = initial_coords[action_idx]
    
    # Calculate trajectory by accumulating displacements
    trajectory = np.zeros((len(action_displacements) + 1, 2))
    trajectory[0] = start_pos
    
    for i, displacement in enumerate(action_displacements):
        trajectory[i + 1] = trajectory[i] + displacement.numpy()

    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot initial shape
    ax.scatter(initial_coords[:, 0], initial_coords[:, 1], 
              c='blue', alpha=0.6, s=50, label='Initial Shape', marker='o')
    ax.plot(initial_coords[:, 0], initial_coords[:, 1], 'b-', alpha=1.0, linewidth=1)
    
    # Plot target shape
    ax.scatter(target_coords[:, 0], target_coords[:, 1], 
              c='red', alpha=0.6, s=50, label='Target Shape', marker='o')
    ax.plot(target_coords[:, 0], target_coords[:, 1], 'r-', alpha=1.0, linewidth=1)
    
    # Plot action trajectory
    color_vals = np.linspace(0, 1, trajectory.shape[0])
    ax.scatter(trajectory[:, 0], trajectory[:, 1], c=color_vals, cmap="inferno", s=50, alpha=0.8)

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title(f'DLO Dataset Sample Visualization\n'
                 f'Manipulated Particle: {action_idx}, Action Steps: {len(action_displacements)}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    ax.set_xlim(-0.05, 0.55)
    ax.set_ylim(-0.3, 0.3)

    plt.tight_layout()
    return fig




def prepare_action(obs_n, action_n):

    action_steps = action_n[1:] / obs_n.shape[1]
    actions_idx = np.tile(action_n[0], (obs_n.shape[0], 1))
    actions = np.tile(action_steps, (obs_n.shape[0], 1))
    actions = np.concatenate([actions_idx, actions], axis=1)

    return actions

def random_horizon_sampling(episode, ep_hor, obs_h_dim, pred_h_dim):
    ep_upper_bound = len(episode) - ep_hor
    action_idx = np.random.randint(0, ep_upper_bound)
    init_obs, action = episode[action_idx]
    final_obs, _ = episode[action_idx + ep_hor - 1]

    init_upper_bound = init_obs.shape[0] - pred_h_dim + 1
    final_upper_bound = final_obs.shape[0] - pred_h_dim + 1
    time_step = np.random.randint(0, init_upper_bound)

    init_obs = init_obs[time_step : time_step + obs_h_dim, :] #(obs_h, n_point, 2)
    action = action[time_step : time_step + pred_h_dim, :] #(act_h, 4)

    time_step = np.random.randint(0, final_upper_bound)
    final_obs = final_obs[time_step : time_step + obs_h_dim, :] #(obs_h, n_point, 2)

    # concatenate init anda target shape
    obs = np.concatenate([init_obs, final_obs], axis=1)

    obs = obs.reshape(obs_h_dim, -1)

    return obs, action


def load_sample(sample):
    dlo_0 = sample["init_shape"].T
    dlo_1 = sample["final_shape"].T
    dlo_0 = dlo_0[:, :2]  # take only x and y coordinates
    dlo_1 = dlo_1[:, :2]  # take only x and y coordinates

    obs = sample["observation"].transpose(0, 2, 1)  # (T, N, 3)
    obs = obs[:, :, :2]  # take only x and y coordinates

    # TODO: extend to a check on the entire observation
    if np.linalg.norm(dlo_0[0, :] - dlo_1[0, :]) > np.linalg.norm(dlo_0[0, :] - dlo_1[-1, :]):
        dlo_1 = np.flip(dlo_1, axis=0)

    action = sample["action"]

    return np.array(dlo_0), np.array(dlo_1), np.array(obs), np.array(action)


def load_and_process_sample(sample=None, disp_range=None, rot_range=None, pred_h_dim=16):
    dlo_0, dlo_1, obs, action = load_sample(sample)
    # dlo_0_n, dlo_1_n, obs_n, action_n, _, _, _ = normalize(dlo_0, dlo_1, obs, action, disp_range, rot_range)
    dlo_0_n, dlo_1_n, obs_n, action_n = dlo_0, dlo_1, obs, action

    # if elements inside action are outside [-1, 1] range, skip
    # if action_n.max() > 1 or action_n.min() < -1:
    #     return None

    #######################

    # prepare observation and action
    action = prepare_action(obs_n, action_n)

    #########################
    # tensors
    norm_obs_tensor = torch.from_numpy(obs_n.copy()).float()
    action_tensor = torch.from_numpy(action.copy()).float()
    return norm_obs_tensor, action_tensor


def load_and_process_samples(file, disp_range=None, rot_range=None, pred_h_dim=16):
    data = pickle.load(open(file, "rb"))
    samples = data["actions_data"]
    normalized_samples = []
    for sample in samples:
        normalized_samples.append(load_and_process_sample(sample, disp_range=disp_range, rot_range=rot_range, pred_h_dim=pred_h_dim))
    return normalized_samples


class DloDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        num_points=16,
        linear_action_range=None,
        rot_action_range=None,
        obs_h_dim=2,
        pred_h_dim=16,
    ):
        super().__init__()

        self.obs_h_dim = obs_h_dim
        self.pred_h_dim = pred_h_dim
        self.ep_hor = 1 # this tells me how many actions from initial shape to final shape, if it is one, i use dlo_0 and dlo_1 of a single action

        self.num_points = num_points
        self.linear_action_range = linear_action_range
        self.rot_action_range = rot_action_range

        assert self.pred_h_dim > self.obs_h_dim, "Prediction horizon must be greater than observation horizon"

        data_files = glob.glob(os.path.join(dataset_path, "*.pkl"))
        print("Found {} files in dataset {}".format(len(data_files), dataset_path))
        self.data_samples = self.preprocess(data_files)

        print(f"Total samples after preprocessing: {len(self.data_samples)}")

    def preprocess(self, data_files):
        data_samples = []
        for file in tqdm(data_files, desc="Processing samples"):
            samples_n = load_and_process_samples(file, self.linear_action_range, self.rot_action_range, self.pred_h_dim)
            data_samples.append(samples_n)
        return data_samples
    
    def set_episode_horizon(self, ep_hor, epoch):
        print(epoch, self.ep_hor)
        self.ep_hor = ep_hor

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        """
        self.data_samples: (episode, action, (obs, action))
        """
        episode = self.data_samples[idx]

        # randomly select actions in the episode given ep_hor, and then timesteps given act_hor and obs_h
        obs, action = random_horizon_sampling(episode, self.ep_hor, self.obs_h_dim, self.pred_h_dim)
        return obs, action


if __name__ == "__main__":
    # Example usage
    DATA_PATH = "/home/mengo/research/dlo_diffusion/train_episodes"

    dataset = DloDataset(DATA_PATH, num_points=51, linear_action_range=0.05, rot_action_range=np.pi / 8, obs_h_dim=1, pred_h_dim=16)

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    for obs, actions in loader:

        # Create visualization
        fig = visualize_sample(obs, actions)
        plt.show()
