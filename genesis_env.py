import os
import genesis as gs
import numpy as np
from env_config import ShapingConfig
from genesis.engine.entities import RigidEntity, MPMEntity
from genesis.engine.entities.rigid_entity import RigidLink
import utils 
from inference import DiffusionInference
from scipy.spatial.transform import Rotation as R
import collections
import time
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))



TARGET_SHAPE = np.array([
    [0.06410741, -0.02362763, 0.00492298],
    [0.0740546,  -0.02464842, 0.00492297],
    [0.08400367, -0.02566472, 0.00492297],
    [0.09395173, -0.02667116, 0.00492296],
    [0.10390425, -0.02766307, 0.00492296],
    [0.11385473, -0.02863455, 0.00492295],
    [0.12381279, -0.02958142, 0.00492295],
    [0.1337676,  -0.03049774, 0.00492294],
    [0.14373229, -0.03138014, 0.00492293],
    [0.15369331, -0.03222329, 0.00492293],
    [0.16366478, -0.03302474, 0.00492292],
    [0.17363334, -0.03378032, 0.00492292],
    [0.18361114, -0.03448846, 0.00492291],
    [0.1935877,  -0.03514656, 0.0049229 ],
    [0.20357094, -0.03575398, 0.0049229 ],
    [0.21355515, -0.03630986, 0.00492289],
    [0.22354268, -0.03681457, 0.00492289],
    [0.23353357, -0.03726904, 0.00492288],
    [0.24352419, -0.03767472, 0.00492288],
    [0.25352036, -0.03803409, 0.00492287],
    [0.26351312, -0.03834968, 0.00492286],
    [0.27351286, -0.03862519, 0.00492286],
    [0.28350731, -0.0388641,  0.00492285],
    [0.29350878, -0.03907092, 0.00492285],
    [0.30350498, -0.03924988, 0.00492284],
    [0.31350653, -0.03940581, 0.00492284],
    [0.32350468, -0.03954325, 0.00492283],
    [0.33350528, -0.0396668,  0.00492283],
    [0.3435052,  -0.03978064, 0.00492282],
    [0.35350483, -0.03988848, 0.00492282],
    [0.36350566, -0.0399933,  0.00492282],
    [0.37350518, -0.04009701, 0.00492281],
    [0.38350557, -0.04020031, 0.00492281],
    [0.39350634, -0.04030218, 0.00492281],
    [0.40350501, -0.04039953, 0.00492281],
    [0.41350824, -0.0404868,  0.0049228 ],
    [0.42350458, -0.0405552,  0.0049228 ],
    [0.43351083, -0.04059236, 0.00492279],
    [0.4435049,  -0.04058129, 0.00492278],
    [0.45351307, -0.04049958, 0.00492276],
    [0.4635039,  -0.04031882, 0.00492273],
    [0.47350838, -0.04000197, 0.00492268],
    [0.48348827, -0.03950491, 0.00492262],
    [0.49347019, -0.03877009, 0.00492255],
    [0.50340972, -0.03773204, 0.00492246],
    [0.5133145,  -0.03630623, 0.00492236],
    [0.52312718, -0.03439945, 0.00492226],
    [0.53281269, -0.0318966,  0.00492219],
    [0.54263561, -0.02883107, 0.00492215],
    [0.55176669, -0.02475351, 0.00492222],
    [0.56055839, -0.02055764, 0.00492228],
])

class ShapingSimplifiedEnv:
    def __init__(self, config: ShapingConfig):
        self.config = config
        gs.init(
            backend=gs.cpu if self.config.simulation.cpu else gs.gpu,
            logging_level="error",
        )
        ########################## create a scene ##########################
        self.scene: gs.Scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.config.simulation.dt,
                substeps=self.config.simulation.substeps,
            ),
            viewer_options=gs.options.ViewerOptions(
                res=self.config.viewer.resolution,
                camera_pos=self.config.viewer.position,
                camera_lookat=self.config.viewer.lookat,
                camera_fov=self.config.viewer.fov,
                refresh_rate=self.config.viewer.refresh_rate,
                max_FPS=self.config.viewer.max_fps,
            ),
            vis_options=gs.options.VisOptions(
                visualize_mpm_boundary=True,
                show_world_frame=True,
            ),
            mpm_options=gs.options.MPMOptions(
                lower_bound=self.config.dlo.mpm_lower_bound,
                upper_bound=self.config.dlo.mpm_upper_bound,
                grid_density=self.config.dlo.mpm_grid_density,
            ),
            show_FPS=self.config.simulation.show_fps,
            show_viewer=self.config.simulation.visualization,
        )

        self.cam = self.scene.add_camera(
            res=self.config.simulation.camera.resolution,
            pos=self.config.simulation.camera.position,
            lookat=self.config.simulation.camera.lookat,
            fov=self.config.simulation.camera.fov,
            GUI=self.config.simulation.camera.gui,
        )

        ########################## entities ##########################
        self.plane = self.scene.add_entity(
            gs.morphs.Plane(),
        )

        self.table = self.scene.add_entity(
            morph=gs.morphs.URDF(
                file=os.path.join(
                    PROJECT_FOLDER, "models/SimpleTable/SimpleTable.urdf"
                ),
                pos=self.config.table.position,
                euler=self.config.table.orientation,
                scale=self.config.table.scale,
                fixed=True,
            ),
            material=gs.materials.Rigid(),
            surface=gs.surfaces.Default(),
        )

        self.dlo: MPMEntity = self.scene.add_entity(
            material=gs.materials.MPM.Elastic(
                E=self.config.dlo.E,  # Determines the squishiness of the rope (very low values act as a sponge)
                nu=self.config.dlo.nu,
                rho=self.config.dlo.rho,
                sampler=self.config.dlo.sampler,
            ),
            morph=gs.morphs.Cylinder(
                height=self.config.dlo.length,
                radius=self.config.dlo.radius,
                pos=self.config.dlo.position,
                euler=self.config.dlo.orientation,
            ),
            surface=gs.surfaces.Default(roughness=2, vis_mode="particle"),
        )
        self.franka: RigidEntity = self.scene.add_entity(
            gs.morphs.MJCF(
                file="xml/franka_emika_panda/panda.xml",
                pos=self.config.franka.position,
                euler=self.config.franka.orientation,
            ),
            material=gs.materials.Rigid(
                friction=self.config.franka.end_effector.friction,
                needs_coup=self.config.franka.end_effector.needs_coup,
                coup_friction=self.config.franka.end_effector.coup_friction,
                sdf_cell_size=self.config.franka.end_effector.sdf_cell_size,
                gravity_compensation=self.config.franka.gravity_compensation,
            ),
        )

        ########################## build ##########################
        self.scene.build()

        self.motors_dof = np.arange(7)
        self.fingers_dof = np.arange(7, 9)
        self.end_effector: RigidLink = self.franka.get_link("hand")

        # Optional: set control gains
        self.franka.set_dofs_kp(
            np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
        )
        self.franka.set_dofs_kv(
            np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
        )
        self.franka.set_dofs_force_range(
            np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
            np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
        )
        self.dlo_init_pos = self.dlo.get_particles()
        self.initial_pose = np.array(
            [
                0.45,
                0.0,
                self.config.table.height
                + self.config.franka.end_effector.offset
                + self.config.franka.end_effector.z_lift,
                0.0,
                0.707,
                0.707,
                0.0,
            ]
        )
        self.model = DiffusionInference(
            checkpoint_path=self.config.inference.model_path,
            device="cuda" if not self.config.simulation.cpu else "cpu",
            noise_steps=100,
        )

        self.obs_horizon = self.model.obs_h_dim

        # Initialize observation buffer
        # Rotate TARGET_SHAPE by 90 degrees counterclockwise around the Z axis
        # Rotate TARGET_SHAPE by 90 degrees counterclockwise around the Z axis
        rotation_matrix = np.array([[0, -1, 0],
                        [1,  0, 0],
                        [0,  0, 1]])
        rotated_target_shape = np.array(TARGET_SHAPE) @ rotation_matrix.T

        # Compute center of rotated target (mean of x and y)
        center_xyz = rotated_target_shape.mean(axis=0)
        rotated_target_shape -= center_xyz

        # Add dlo position
        rotated_target_shape += self.config.dlo.position

        self.target = rotated_target_shape
        obs = self.get_obs()
        self.obs_deque = collections.deque(
            [obs] * self.obs_horizon, maxlen=self.obs_horizon
        )

        self.ready_to_plot = False
        self.current_pred_action = None

    def plot(self):
        plt.ion()  # Enable interactive mode

        if not hasattr(self, "_fig") or self._fig is None:
            self._fig, self._ax = plt.subplots(figsize=(6, 6))
        else:
            self._ax.clear()

        current_dlo_shape = utils.get_skeleton(
            self.dlo.get_particles(),
            downsample_number=self.config.dlo.number_of_particles,
            average_number=self.config.dlo.particles_smoothing,
        )
        target_shape = self.target[:, :2]
        pred_action = self.current_pred_action[:, 1:3]

        self._ax.plot(
            current_dlo_shape[:, 0],
            current_dlo_shape[:, 1],
            "o-",
            label="DLO",
            linewidth=2,
            markersize=4,
        )
        self._ax.plot(
            target_shape[:, 0],
            target_shape[:, 1],
            "o-",
            label="Target",
            linewidth=2,
            markersize=4,
        )
        self._ax.scatter(
            current_dlo_shape[0, 0], current_dlo_shape[0, 1], marker="x", s=60, c="b"
        )
        self._ax.scatter(
            target_shape[0, 0], target_shape[0, 1], marker="x", s=60, c="r"
        )

        if pred_action is not None:
            pred_pt = []
            pt = self.end_effector.get_pos().cpu().numpy()[:2]
            for delta in pred_action:
                pt += delta
                pred_pt.append(pt.copy())
            pred_pt = np.array(pred_pt)
            self._ax.plot(
                pred_pt[:, 0],
                pred_pt[:, 1],
                "^-",
                label="Predicted Action",
                linewidth=2,
                markersize=4,
            )

        self._ax.set_xlabel("X")
        self._ax.set_ylabel("Y")
        self._ax.set_title("DLO vs Target in XY Plane")
        self._ax.legend()
        self._ax.axis("equal")
        self._ax.grid(True)

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    def _step(self):
        start_time = time.time()

        self.scene.step()
        end_time = time.time()
        self.real_time_factor = self.config.simulation.dt / (end_time - start_time)

        if self.config.simulation.camera.record:
            self.cam.render()

        if self.config.inference.plot and self.ready_to_plot:
            self.plot()

        if self.config.simulation.show_real_time_factor:
            print(f"Real-time factor: {self.real_time_factor:.2f}")

    def reset_robot_pose(self):
        # Place robot above centre of the rope
        skeleton = utils.get_skeleton(
            self.dlo.get_particles(),
            downsample_number=self.config.dlo.number_of_particles,
            average_number=self.config.dlo.particles_smoothing,
        )
        skeleton_centre = np.array([np.mean(skeleton[:, 0]), np.mean(skeleton[:, 1])])

        target_pos = [
            skeleton_centre[0],
            skeleton_centre[1],
            self.config.table.height
            + self.config.franka.end_effector.offset
            + self.config.franka.end_effector.z_lift,
        ]
        target_quat = self.initial_pose[3:7]  # Use the initial pose's quaternion

        qpos = self.franka.inverse_kinematics(
            link=self.end_effector,
            pos=target_pos,
            quat=target_quat,
        )
        qpos[-2:] = (
            self.config.franka.end_effector.gripper_open_position
        )  # Open gripper at the start

        self.franka.set_qpos(qpos)

    def reset_dlo_pose(self):
        self.dlo.set_pos(self.dlo._sim.cur_substep_local, self.dlo_init_pos)

    def reset_episode(self):
        """Reset the environment for a new episode."""
        self.scene.clear_debug_objects()

        # Choose new target
        # self.target = np.array(TARGET_SHAPE) + self.config.dlo.position
        utils.draw_skeleton(self.target, self.scene, self.config.dlo.radius)

        # Reset robot pose
        self.reset_robot_pose()

        # Reset DLO pose
        self.reset_dlo_pose()

        self._step()

    def reset_action(self):
        """Reset the environment for a new action."""
        self.scene.clear_debug_objects()

        # Redraw the target shape
        utils.draw_skeleton(self.target, self.scene, self.config.dlo.radius)

        # Reset robot pose
        self.reset_robot_pose()

        self._step()

    def get_obs(self):
        obs_dlo = utils.get_skeleton(
            self.dlo.get_particles(),
            downsample_number=self.config.dlo.number_of_particles,
            average_number=self.config.dlo.particles_smoothing,
        )[:, :2]  # Only x and y coordinates of the DLO
        obs_target = self.target[:, :2]  # Only x and y coordinates of the target
        obs_dlo = np.array(obs_dlo).flatten()
        obs_target = np.array(obs_target).flatten()
        return np.concatenate([obs_dlo, obs_target])

    def draw_trajectory(self, traj):
        """Draw the trajectory of the end-effector."""
        target_pos = self.end_effector.get_pos().cpu().numpy()
        target_pos[2] -= self.config.franka.end_effector.offset

        for i, action in enumerate(traj):
            # Red to blue gradient
            t = i / (len(action) - 1)
            color = [1.0 - t, 0.0, t, 1.0]

            dx, dy = action[1:3]

            target_pos += np.array([dx, dy, 0.0])

            self.scene.draw_debug_sphere(
                target_pos,
                color=color,
                radius=0.001,
            )

    def execute_action(
        self, target_pos, target_quat, gripper="open", path_period=1.0, tolerance=1e-7
    ):
        qpos = self.franka.inverse_kinematics(
            link=self.end_effector,
            pos=target_pos,
            quat=target_quat,
        )
        if gripper == "open":
            force_control = [0.0, 0.0]
            # qpos[-2:] = self.config.franka.end_effector.gripper_open_position
        elif gripper == "close":
            force_control = [-1.0, -1.0]
            # qpos[-2:] = self.config.franka.end_effector.gripper_close_position

        path = self.franka.plan_path(
            qpos_goal=qpos,
            num_waypoints=int(path_period // self.config.simulation.dt),
            ignore_collision=True,  # Otherwise cannot grasp in a good way the rope
        )

        # Control the robot along the path
        for p in path:
            self.franka.control_dofs_position(p[:-2], self.motors_dof)
            self.franka.control_dofs_force(force_control, self.fingers_dof)

            self._step()

            # Check if the robot has reached the target position
            if (
                np.linalg.norm(
                    qpos.cpu().numpy() - self.franka.get_qpos().cpu().numpy()
                )
                < tolerance
            ):
                break

    def grasp(self, pred_idx):
        # Grasp
        pred_idx = int(pred_idx)

        skeleton = utils.get_skeleton(
            self.dlo.get_particles(),
            downsample_number=self.config.dlo.number_of_particles,
            average_number=self.config.dlo.particles_smoothing,
        )

        if self.config.inference.interactive:
            pred_idx = int(
                input(f"Select particle index to grasp (0-{len(skeleton)-1}): ")
            )

        target_pos, target_quat = utils.compute_pose_from_paticle_index(
            skeleton,
            pred_idx,
            self.config.franka.end_effector.rot_offset,
            self.config.franka.end_effector.offset,
        )

        self.execute_action(
            target_pos=target_pos,
            target_quat=target_quat,
            gripper="open",
            path_period=1.0,
        )

        self.execute_action(
            target_pos=target_pos,
            target_quat=target_quat,
            gripper="close",
            path_period=0.5,
        )

    def execute_trajectory(self, traj):
        # Execute action
        for action in traj:
            dx, dy, dt = action[1:4]

            current_pos = self.end_effector.get_pos().cpu().numpy()
            current_quat = self.end_effector.get_quat().cpu().numpy()

            current_R = (R.from_quat(current_quat)).as_matrix()
            delta_R = R.from_euler("xyz", [dt, 0.0, 0.0]).as_matrix()
            target_R = current_R @ delta_R

            target_pos = current_pos + np.array([dx, dy, 0.0])
            target_quat = (R.from_matrix(target_R)).as_quat()

            self.execute_action(
                target_pos=target_pos,
                target_quat=target_quat,
                gripper="close",
                path_period=0.2,
            )

        # Release

        self.execute_action(
            target_pos=target_pos,
            target_quat=target_quat,
            gripper="open",
            path_period=0.5,
        )

    def run(self):
        if self.config.simulation.camera.record:
            self.cam.start_recording()

        for _ in tqdm(range(self.config.inference.n_episodes), desc="Episodes"):
            self.reset_episode()
            for _ in tqdm(range(self.config.inference.n_actions), desc="Actions"):
                self.reset_action()
                obs = self.get_obs()
                print("obs.shape", obs.shape)
                self.obs_deque.append(obs)
                obs = np.stack(self.obs_deque)
                obs = obs.reshape(self.obs_horizon, -1)

                pred_action = self.model.run(obs)
                print("pred_action.shape", pred_action.shape)

                self.grasp(int(pred_action[0, 0]))
                self.ready_to_plot = True
                for _ in tqdm(range(2), desc="Steps"):
                    obs = self.get_obs()

                    self.obs_deque.append(obs)
                    obs = np.stack(self.obs_deque)
                    obs = obs.reshape(self.obs_horizon, -1)
                    pred_action = self.model.run(obs)
                    self.current_pred_action = pred_action
                    self.draw_trajectory(pred_action)
                    self.execute_trajectory(pred_action)

        if self.config.simulation.camera.record:
            self.cam.stop_recording(save_to_filename="video.mp4", fps=60)
        if self.config.inference.plot:
            plt.ioff()
            plt.close("all")


def main():
    parser = argparse.ArgumentParser(description="Teleop Push Data Generator")
    parser.add_argument(
        "--cfg",
        type=str,
        default="env_config.yaml",
        help="Path to the configuration file",
    )
    parser.add_argument("-v", "--vis", action="store_true")
    parser.add_argument("-g", "--gui", action="store_true", help="Enable GUI mode")
    parser.add_argument(
        "-c", "--cpu", action="store_true", help="Run on CPU instead of GPU"
    )
    parser.add_argument(
        "-f", "--show_fps", action="store_true", help="Show FPS in the viewer"
    )
    parser.add_argument(
        "-e", "--n_episodes", type=int, default=1, help="Number of episodes to run"
    )
    parser.add_argument(
        "-a", "--n_actions", type=int, default=1, help="Number of actions per episode"
    )
    parser.add_argument(
        "-r", "--record", action="store_true", help="Record the simulation"
    )
    parser.add_argument("-p", "--plot", action="store_true", help="Plot the results")
    parser.add_argument(
        "-i", "--interactive", action="store_true", help="Run in interactive mode"
    )
    args = parser.parse_args()

    config = ShapingConfig.from_yaml_and_args(args.cfg, args)
    env = ShapingSimplifiedEnv(config)
    env.run()


if __name__ == "__main__":
    main()
