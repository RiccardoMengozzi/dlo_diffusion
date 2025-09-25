import os, pickle, glob
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from model.normalize import normalize


def load_sample(data_dict):

    dlo_0 = data_dict["dlo_0"]
    dlo_1 = data_dict["dlo_1"]
    dlo_0 = dlo_0[:, :2]  # take only x and y coordinates
    dlo_1 = dlo_1[:, :2]  # take only x and y coordinates

    if np.linalg.norm(dlo_0[0, :] - dlo_1[0, :]) > np.linalg.norm(dlo_0[0, :] - dlo_1[-1, :]):
        dlo_1 = np.flip(dlo_1, axis=0)

    action_traj = data_dict["action_traj"]

    return np.array(dlo_0), np.array(dlo_1), np.array(action_traj)


def load_and_process_sample(file, disp_range=None, rot_range=None, pred_h_dim=16):
    data = pickle.load(open(file, "rb"))

    list_obs, list_actions = [], []
    for step, sample in data.items():

        dlo_0, dlo_1, action = load_sample(sample)
        dlo_0_n, dlo_1_n, action_n, _, _, _ = normalize(dlo_0, dlo_1, action, disp_range, rot_range)

        # if elements inside action are outside [-1, 1] range, skip
        if action_n.max() > 1 or action_n.min() < -1:
            print(np.linalg.norm(action[1:3]), action_n.max(), action_n.min())
            return None

        # prepare action given pred_h_dim
        idx = np.linspace(0, action_n.shape[0] - 1, pred_h_dim, dtype=int)
        action_horizon = action_n[idx, :]
        actions_tensor = torch.from_numpy(action_horizon.copy()).float()

        # prepare obs given pred_h_dim
        norm_obs = np.concatenate([dlo_0_n, dlo_1_n], axis=0)
        norm_obs = norm_obs.reshape(1, norm_obs.shape[0] * 2)
        norm_obs_tensor = torch.from_numpy(norm_obs.copy()).float()

        # ready
        list_obs.append(norm_obs_tensor)
        list_actions.append(actions_tensor)

    return list_obs, list_actions


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
            rv = load_and_process_sample(file, self.linear_action_range, self.rot_action_range, self.pred_h_dim)
            if rv is None:
                print(f"Skipping file {file} due to out-of-range actions")
                continue

            obs, actions = rv
            for ob, ac in zip(obs, actions):
                data_samples.append((ob, ac))

        return data_samples

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        """
        obs shape: torch.Size([B, obs_h_dim, obs_dim])
        action shape: torch.Size([B, pred_h_dim, action_dim])
        """
        obs, action = self.data_samples[idx]
        return obs, action


if __name__ == "__main__":
    # Example usage
    DATA_PATH = "/home/lar/dev25/DLO_DIFFUSION/DATA/dataset_11_09/val_spline"

    dataset = DloDataset(DATA_PATH, num_points=51, linear_action_range=0.0707, rot_action_range=np.pi / 8)

    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)

    for obs, actions in loader:
        print(obs.shape, actions.shape)
        break