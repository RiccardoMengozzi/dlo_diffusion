import os, pickle, glob
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

from tqdm import tqdm
from normalize import normalize


def prepare_obs_action(obs_n, dlo_1_n, action_n):

    action_steps = action_n[1:] / obs_n.shape[1]
    actions_idx = np.tile(action_n[0], (obs_n.shape[0], 1))
    actions = np.tile(action_steps, (obs_n.shape[0], 1))
    actions = np.concatenate([actions_idx, actions], axis=1)

    obs_target = np.tile(dlo_1_n, (obs_n.shape[0], 1, 1))
    norm_obs = np.concatenate([obs_n, obs_target], axis=1)

    # flatten the observation
    norm_obs = norm_obs.reshape(norm_obs.shape[0], -1)

    return norm_obs, actions


def random_horizon_sampling(obs, action, obs_h_dim, pred_h_dim):
    upper_bound = obs.shape[0] - pred_h_dim + 1
    time_step = np.random.randint(0, upper_bound)
    obs = obs[time_step : time_step + obs_h_dim, :]
    action = action[time_step : time_step + pred_h_dim, :]

    return obs, action


def load_sample(file):
    data = pickle.load(open(file, "rb"))

    dlo_0 = data["init_shape"].T
    dlo_1 = data["final_shape"].T
    dlo_0 = dlo_0[:, :2]  # take only x and y coordinates
    dlo_1 = dlo_1[:, :2]  # take only x and y coordinates

    obs = data["observation"].transpose(0, 2, 1)  # (T, N, 3)
    obs = obs[:, :, :2]  # take only x and y coordinates

    # TODO: extend to a check on the entire observation
    if np.linalg.norm(dlo_0[0, :] - dlo_1[0, :]) > np.linalg.norm(dlo_0[0, :] - dlo_1[-1, :]):
        dlo_1 = np.flip(dlo_1, axis=0)

    action = data["action"]

    return np.array(dlo_0), np.array(dlo_1), np.array(obs), np.array(action)


def load_and_process_sample(file, disp_range=None, rot_range=None):
    dlo_0, dlo_1, obs, action = load_sample(file)
    _, dlo_1_n, obs_n, action_n, _, _, _ = normalize(dlo_0, dlo_1, obs, action, disp_range, rot_range)

    # if elements inside action are outside [-1, 1] range, skip
    if action_n.max() > 1 or action_n.min() < -1:
        return None

    #######################

    # prepare observation and action
    norm_obs, actions = prepare_obs_action(obs_n, dlo_1_n, action_n)

    #########################
    # tensors
    norm_obs_tensor = torch.from_numpy(norm_obs.copy()).float()
    actions_tensor = torch.from_numpy(actions.copy()).float()
    return norm_obs_tensor, actions_tensor


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
            rv = load_and_process_sample(file, self.linear_action_range, self.rot_action_range)
            if rv is None:
                continue

            obs, _ = rv
            upper_bound = obs.shape[0] - self.pred_h_dim + 1

            if upper_bound <= 0:
                continue

            data_samples.append(rv)

        return data_samples

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        """
        obs shape: torch.Size([B, obs_h_dim, obs_dim])
        action shape: torch.Size([B, pred_h_dim, action_dim])
        """

        obs, action = self.data_samples[idx]

        # randomly select a time step for the observation and action
        obs, action = random_horizon_sampling(obs, action, self.obs_h_dim, self.pred_h_dim)

        return obs, action


if __name__ == "__main__":
    # Example usage
    DATA_PATH = "/home/mengo/Research/LLM_DOM/diffusion/train3"

    dataset = DloDataset(DATA_PATH, num_points=51, linear_action_range=0.05, rot_action_range=np.pi / 8)

    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)

    for obs, actions in loader:
        print(obs.shape, actions.shape)

        break