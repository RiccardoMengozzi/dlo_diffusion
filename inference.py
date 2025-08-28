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


DLO_0_N_TEST = np.array([[-0.24576193,  0.00006841],
        [-0.23842283, -0.00002251],
        [-0.22844487,  0.00012381],
        [-0.21831221, -0.00013607],
        [-0.20829912,  0.00004509],
        [-0.19852597,  0.00002071],
        [-0.18863376, -0.00013362],
        [-0.17862239,  0.00013134],
        [-0.16849192, -0.00012976],
        [-0.1584807 ,  0.00004854],
        [-0.1487078 ,  0.00002425],
        [-0.13881573, -0.00013011],
        [-0.1288045 ,  0.00013485],
        [-0.11915079,  0.00005128],
        [-0.10937787,  0.00002708],
        [-0.09948576, -0.00012734],
        [-0.08947451,  0.00013762],
        [-0.07934402, -0.00012354],
        [-0.06933276,  0.00005479],
        [-0.05955984,  0.0000305 ],
        [-0.04966774, -0.00012383],
        [-0.03965648,  0.00014113],
        [-0.02952601, -0.00012003],
        [-0.01951475,  0.00005839],
        [-0.00974183,  0.0000341 ],
        [ 0.00015027, -0.00012026],
        [ 0.009804  , -0.0001172 ],
        [ 0.01981526,  0.00006107],
        [ 0.02958817,  0.00003687],
        [ 0.03948028, -0.00011755],
        [ 0.04949154,  0.00014741],
        [ 0.05962201, -0.00011369],
        [ 0.06963327,  0.00006461],
        [ 0.07940618,  0.00004038],
        [ 0.08929829, -0.00011398],
        [ 0.09930954,  0.00015092],
        [ 0.10944003, -0.00011024],
        [ 0.1194513 ,  0.00006809],
        [ 0.1286283 , -0.00011127],
        [ 0.13863954,  0.00015369],
        [ 0.14876996, -0.00010747],
        [ 0.15878121,  0.00007086],
        [ 0.16855409,  0.0000466 ],
        [ 0.17844616, -0.00010776],
        [ 0.18845755,  0.0001572 ],
        [ 0.19858809, -0.00010396],
        [ 0.20859942,  0.0000744 ],
        [ 0.21837217,  0.00004773],
        [ 0.22824936, -0.00011361],
        [ 0.23792662,  0.0001189 ],
        [ 0.24565348, -0.00008682]])

DLO_1_N_TEST =  np.array([[-0.24070267,  0.00292686],
        [-0.23132242, -0.00052248],
        [-0.22193218, -0.00398182],
        [-0.21255194, -0.00743116],
        [-0.20316169, -0.0108705 ],
        [-0.19375145, -0.01426983],
        [-0.18433122, -0.01762917],
        [-0.17489099, -0.02091851],
        [-0.16541076, -0.02410784],
        [-0.15589055, -0.02715717],
        [-0.14702029, -0.03084654],
        [-0.1373801 , -0.03350586],
        [-0.12681004, -0.03437512],
        [-0.11684998, -0.03519442],
        [-0.10684997, -0.03527371],
        [-0.09686001, -0.03472301],
        [-0.08692009, -0.03365231],
        [-0.0770302 , -0.03214161],
        [-0.06721033, -0.03027092],
        [-0.05744048, -0.02810023],
        [-0.04774065, -0.02569955],
        [-0.03807083, -0.02311887],
        [-0.02845102, -0.02039819],
        [-0.01886122, -0.01756751],
        [-0.00929143, -0.01466684],
        [ 0.00026837, -0.01172617],
        [ 0.00981816, -0.00875549],
        [ 0.01936795, -0.00578482],
        [ 0.02891774, -0.00282415],
        [ 0.03847753,  0.00010653],
        [ 0.04804733,  0.0030072 ],
        [ 0.05763713,  0.00584788],
        [ 0.06723693,  0.00864855],
        [ 0.07685674,  0.01137923],
        [ 0.08649655,  0.01404991],
        [ 0.09614637,  0.01666059],
        [ 0.10581619,  0.01920127],
        [ 0.11551601,  0.02167195],
        [ 0.12521584,  0.02408264],
        [ 0.13493568,  0.02642332],
        [ 0.14467552,  0.02870401],
        [ 0.15442536,  0.03092469],
        [ 0.16418521,  0.03309538],
        [ 0.17396506,  0.03520607],
        [ 0.18374492,  0.03727676],
        [ 0.19353477,  0.03930745],
        [ 0.20333463,  0.04130814],
        [ 0.21314449,  0.04327883],
        [ 0.22294435,  0.04522952],
        [ 0.23275422,  0.04717021],
        [ 0.24257408,  0.0491009 ]])





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
    DATA_PATH = os.path.join(MAIN_DIR, "DATA/train")
    CHECKPOINT_PATH = os.path.join(MAIN_DIR, "checkpoints/diffusion_dummy-cr4zd6r5_best.pt")
    NOISE_STEPS = 100

    dlo_diff = DiffusionInference(CHECKPOINT_PATH, device="cuda", noise_steps=NOISE_STEPS)

    ################################

    files = os.listdir(DATA_PATH)
    files = [os.path.join(DATA_PATH, f) for f in files if f.endswith(".pkl")]
    print(f"Found {len(files)} files in {DATA_PATH}")

    for file in files:

        file_path = os.path.join(DATA_PATH, file)

        # LOAD
        dlo_0_plot, dlo_1_plot, obs, act = load_sample(file_path)


        dlo_0_plot = DLO_0_N_TEST
        dlo_1_plot = DLO_1_N_TEST

        dlo_0 = dlo_0_plot.reshape(1, -1)
        dlo_1 = dlo_1_plot.reshape(1, -1)
        print(f"dlo0 shape = {dlo_0.shape}, dlo1 shape = {dlo_1.shape}")
        obs = np.concatenate([dlo_0, dlo_1], axis=1)
        print(f"obs shape = {obs.shape}")


        # Run inference
        pred_action = dlo_diff.run(obs)


        fig = plt.figure(figsize=(12, 6))

        plt.plot(dlo_0_plot[:, 0], dlo_0_plot[:, 1], label="dlo_0_original", color="red", alpha=0.5)
        plt.plot(dlo_1_plot[:, 0], dlo_1_plot[:, 1], label="dlo_1_original", color="green", alpha=0.5)
        plt.scatter(dlo_0_plot[0, 0], dlo_0_plot[0, 1], c="red")
        plt.scatter(dlo_1_plot[0, 0], dlo_1_plot[0, 1], c="green")


        pred_action_accumulated = []
        accumulator = dlo_0_plot[int(pred_action[0,0])]
        for action in pred_action:
            accumulator += np.array(action[1:3])
            pred_action_accumulated.append(accumulator.copy())

        pred_action_accumulated = np.array(pred_action_accumulated)


        plt.plot(pred_action_accumulated[:, 0], pred_action_accumulated[:, 1], label="pred_action", marker="o")


        plt.tight_layout()
        plt.legend()
        plt.axis("equal")
        plt.show()


