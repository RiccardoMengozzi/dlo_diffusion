import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

from conditional_1d_unet import ConditionalUnet1D
from diffusers.schedulers import DDPMScheduler

import normalize
from dataset import load_sample

np.set_printoptions(precision=8, suppress=True, linewidth=100, threshold=1000)

class DiffusionInference:
    def __init__(self, checkpoint_path, device="cpu", noise_steps=100):

        state = torch.load(checkpoint_path)

        self.disp_scale = state["scale_disp"]
        self.angle_scale = state["scale_rot"]
        self.num_points = state["num_points"]

        self.pred_horizon = state["pred_h_dim"]
        self.obs_dim = state["obs_dim"]
        self.obs_h_dim = state["obs_h_dim"]
        self.action_dim = state["action_dim"]
        self.device = device

        print("obs_dim:", self.obs_dim)

        self.noise_steps = noise_steps

        # Build diffusion components
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.noise_steps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

        self.model = ConditionalUnet1D(
            input_dim=state["action_dim"], global_cond_dim=state["obs_dim"] * state["obs_h_dim"]
        )
        self.model.load_state_dict(state["model"])
        self.model.to(self.device)
        self.model.eval()

    def run_denoise_action(self, obs_horizon):

        norm_obs_tensor = torch.from_numpy(obs_horizon.copy()).float().unsqueeze(0).to(self.device)

        obs_cond = norm_obs_tensor.flatten(start_dim=1).to(self.device)

        # initialize action from Gaussian noise
        naction = torch.randn((1, self.pred_horizon, self.action_dim), device=self.device)

        # init scheduler
        self.noise_scheduler.set_timesteps(self.noise_steps)

        list_actions = []
        for k in tqdm(self.noise_scheduler.timesteps):
            # predict noise
            noise_pred = self.model(sample=naction, timestep=k, global_cond=obs_cond)

            # inverse diffusion step (remove noise)
            naction = self.noise_scheduler.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample

            list_actions.append(naction.squeeze().detach().cpu().numpy())

        return np.stack(list_actions, axis=0)

    def denormalize_action(self, dlo, action, cs0, csR, rot_check_flag=False):
        action_dn = normalize.denormalize_action_horizon(
            dlo, action, cs0, csR, self.disp_scale, self.angle_scale, rot_check_flag=rot_check_flag
        )
        return action_dn

    def run(self, obs):
        dlo_0 = obs[:, :self.obs_dim // 2].reshape(self.obs_h_dim, self.num_points, -1)
        dlo_1 = obs[:, self.obs_dim // 2:].reshape(self.obs_h_dim, self.num_points, -1)
        cs0, csR = normalize.compute_cs0_csR(dlo_0[-1])
        rot_check_flag = np.array([False] * self.obs_h_dim, dtype=bool)
        dlo_0_n = np.zeros_like(dlo_0)
        dlo_1_n = np.zeros_like(dlo_1)
        for i in range(len(dlo_0)):
            dlo_0[i] = normalize.normalize_dlo(dlo_0[i], cs0, csR)
            dlo_1[i] = normalize.normalize_dlo(dlo_1[i], cs0, csR)
            dlo_0_n[i], dlo_1_n[i], rot_check_flag[i] = normalize.check_rot_and_flip(dlo_0[i], dlo_1[i])

        dlo_0_n = dlo_0_n.reshape(self.obs_h_dim, -1)
        dlo_1_n = dlo_1_n.reshape(self.obs_h_dim, -1)
        obs = np.concatenate([dlo_0_n, dlo_1_n], axis=1)
        ###################################################################
        pred_actions = self.run_denoise_action(obs)

        pred_action = pred_actions[-1]  # take the last action from the denoised actions

        ############################
        pred_action_dn = self.denormalize_action(
            dlo_0[-1], pred_action, cs0, csR, rot_check_flag=rot_check_flag[-1]
        )

        return pred_action_dn



if __name__ == "__main__":

    MAIN_DIR = os.path.dirname(__file__)
    DATA_PATH = os.path.join(MAIN_DIR, "train3")
    CHECKPOINT_PATH = os.path.join(MAIN_DIR, "diffusion_smooth-field-4_best.pt")
    NOISE_STEPS = 100

    dlo_diff = DiffusionInference(CHECKPOINT_PATH, device="cuda", noise_steps=NOISE_STEPS)

    ################################

    files = os.listdir(DATA_PATH)
    files = [os.path.join(DATA_PATH, f) for f in files if f.endswith(".pkl")]
    print(f"Found {len(files)} files in {DATA_PATH}")

    for file in files:

        file_path = os.path.join(DATA_PATH, file)

        # LOAD
        dlo_0, dlo_1, obs, act = load_sample(file_path)
        dlo_0 = np.array([dlo_0, dlo_0])
        dlo_1 = np.array([dlo_1, dlo_1])


        dlo_0 = dlo_0.reshape(2, -1)
        dlo_1 = dlo_1.reshape(2, -1)
        obs = np.concatenate([dlo_0, dlo_1], axis=1)

        print(obs.shape)

        # Run inference
        dlo_diff.run(obs)

        break