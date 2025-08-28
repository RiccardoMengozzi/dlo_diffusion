import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

from conditional_1d_unet import ConditionalUnet1D
from diffusers.schedulers import DDPMScheduler

from normalize import normalize, denormalize_dlo, denormalize_action_horizon, convert_action_horizon_to_absolute
from dataset import load_sample, prepare_obs_action, random_horizon_sampling




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
        print("norm_obs_tensor shape = ", norm_obs_tensor.shape)

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

    def denormalize_action_absolute(self, dlo, action_horizon, cs0, csR, rot_check_flag=False):
        act = denormalize_action_horizon(
            dlo,
            action_horizon,
            cs0,
            csR,
            disp_scale=self.disp_scale,
            angle_scale=self.angle_scale,
            rot_check_flag=rot_check_flag,
        )

        act_pos, act_rot = convert_action_horizon_to_absolute(dlo, act)

        return act_pos, act_rot

    def from_observation_to_dlos(self, obs, cs0, csR, rot_check_flag=False):
        obs_last = obs[-1, :].reshape(-1)
        dlo_0_n = obs_last[: self.num_points * 2].reshape(self.num_points, 2)
        dlo_1_n = obs_last[self.num_points * 2 :].reshape(self.num_points, 2)

        dlo_0_ok = denormalize_dlo(dlo_0_n, cs0, csR, rot_check_flag)
        dlo_1_ok = denormalize_dlo(dlo_1_n, cs0, csR, rot_check_flag)

        return dlo_0_ok, dlo_1_ok

    def run(self, dlo_0, dlo_1, obs, act):
        dlo_0_n, dlo_1_n, obs_n, action_n, cs0, csR, rot_check_flag = normalize(
            dlo_0, dlo_1, obs, act, disp_scale=self.disp_scale, angle_scale=self.angle_scale
        )
        print("obs shape = ", obs.shape)

        obs, action = prepare_obs_action(obs_n, dlo_1_n, action_n)
        print(obs.shape)
        obs_horizon, action_horizon = random_horizon_sampling(
            obs, action, obs_h_dim=self.obs_h_dim, pred_h_dim=self.pred_horizon
        )
        print("after sampling: ", obs_horizon.shape, action_horizon.shape)

        ###################################################################
        dlo_0_ok, dlo_1_ok = self.from_observation_to_dlos(obs_horizon, cs0, csR, rot_check_flag)
        print("dlo_0:", dlo_0_ok.shape, "dlo_1:", dlo_1_ok.shape)

        action_original_pos, action_original_rot = self.denormalize_action_absolute(
            dlo_0_ok, action, cs0, csR, rot_check_flag=rot_check_flag
        )

        action_horizon_orig_abs_pos, action_horizon_orig_abs_rot = self.denormalize_action_absolute(
            dlo_0_ok, action_horizon, cs0, csR, rot_check_flag=rot_check_flag
        )

        #### TEST ####
        dlo_0_ok = DLO_0_N_TEST
        dlo_1_ok = DLO_1_N_TEST
        obs_horizon = np.concatenate([DLO_0_N_TEST, DLO_1_N_TEST]).reshape(1, -1)
        print("obs_horizon shape = ", obs_horizon.shape)


        ###################################################################
        print("obs_horizon:", obs_horizon.shape)
        pred_actions = self.run_denoise_action(obs_horizon)

        pred_action = pred_actions[-1]  # take the last action from the denoised actions

        ############################
        pred_action_pos, pred_action_rot = self.denormalize_action_absolute(
            dlo_0_ok, pred_action, cs0, csR, rot_check_flag=rot_check_flag
        )

        print("REAL ACTION = ", act / obs.shape[0])
        print("PRED ACTION = ", pred_action_pos, pred_action_rot)

        if False:
            fig = plt.figure(figsize=(12, 6))

            plt.plot(dlo_0_ok[:, 0], dlo_0_ok[:, 1], label="dlo_0_ok", color="blue")
            plt.plot(dlo_1_ok[:, 0], dlo_1_ok[:, 1], label="dlo_1_ok", color="orange")
            plt.scatter(dlo_0_ok[0, 0], dlo_0_ok[0, 1], c="blue")
            plt.scatter(dlo_1_ok[0, 0], dlo_1_ok[0, 1], c="orange")

            plt.plot(dlo_0[:, 0], dlo_0[:, 1], label="dlo_0_original", color="red", alpha=0.5)
            plt.plot(dlo_1[:, 0], dlo_1[:, 1], label="dlo_1_original", color="green", alpha=0.5)
            plt.scatter(dlo_0[0, 0], dlo_0[0, 1], c="red")
            plt.scatter(dlo_1[0, 0], dlo_1[0, 1], c="green")

            plt.plot(action_original_pos[:, 0], action_original_pos[:, 1], label="action_original", marker="o")
            plt.plot(
                action_horizon_orig_abs_pos[:, 0],
                action_horizon_orig_abs_pos[:, 1],
                label="gt_action_horizon",
                marker="o",
            )
            plt.plot(pred_action_pos[:, 0], pred_action_pos[:, 1], label="pred_action", marker="o")

            plt.tight_layout()
            plt.legend()
            plt.axis("equal")
            plt.show()

        if True:
            from matplotlib.widgets import Slider

            # Plot initialization
            fig, ax = plt.subplots(figsize=(12, 6))
            plt.subplots_adjust(bottom=0.25)

            # Plot the static DLO curves
            (dlo0_line,) = ax.plot(dlo_0_ok[:, 0], dlo_0_ok[:, 1], label="dlo_0_ok", color="blue")
            (dlo1_line,) = ax.plot(dlo_1_ok[:, 0], dlo_1_ok[:, 1], label="dlo_1_ok", color="orange")

            # Initial scatter plot
            initial_i = 0
            try:
                action_pos, action_rot = self.denormalize_action_absolute(
                    dlo_0_ok, pred_actions[initial_i], cs0, csR, rot_check_flag=rot_check_flag
                )
                # Use a scalar value to map colors (e.g., index, or norm of displacement)
                color_vals = np.linspace(0, 1, action_pos.shape[0])
                scatter = ax.scatter(action_pos[:, 0], action_pos[:, 1], c=color_vals, cmap="inferno", s=50, alpha=0.8)
            except Exception as e:
                print(f"Error denormalizing action {initial_i}: {e}")
                scatter = ax.scatter([], [], c=[], cmap="inferno", s=50, alpha=0.8)

            # Add slider
            ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
            slider = Slider(ax_slider, "Action Index", 0, len(pred_actions) - 1, valinit=initial_i, valstep=1)

            # Update function
            def update(val):
                i = int(slider.val)
                try:
                    action_pos, action_rot = self.denormalize_action_absolute(
                        dlo_0_ok, pred_actions[i], cs0, csR, rot_check_flag=rot_check_flag
                    )
                    scatter.set_offsets(action_pos[:, :2])  # x, y positions
                except Exception as e:
                    print(f"Error denormalizing action {i}: {e}")
                    scatter.set_offsets([])

                fig.canvas.draw_idle()

            slider.on_changed(update)

            # Final touches
            ax.axis("equal")
            ax.legend()
            plt.tight_layout()
            plt.show()


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
        dlo_0, dlo_1, obs, act = load_sample(file_path)

        # Run inference
        dlo_diff.run(dlo_0, dlo_1, obs, act)

        