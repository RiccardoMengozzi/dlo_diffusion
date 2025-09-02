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



U_SHAPE = np.array([
    [0.5000227,  0.23180239,  0.7030916 ],
    [0.49989584, 0.22447005,  0.70354724],
    [0.49945357, 0.21452278,  0.7035685 ],
    [0.49948218, 0.20441478,  0.703463  ],
    [0.49924994, 0.1944266,   0.70341164],
    [0.49953243, 0.1846698,   0.7036865 ],
    [0.5003945,  0.17481326,  0.70362806],
    [0.50135416, 0.16482817,  0.7035512 ],
    [0.5033843,  0.15486442,  0.70344424],
    [0.5055199,  0.14501947,  0.7033917 ],
    [0.50829244, 0.13555898,  0.7036578 ],
    [0.5116568,  0.12616804,  0.7035981 ],
    [0.51515365, 0.11670154,  0.70352834],
    [0.51924855, 0.10788424,  0.70338637],
    [0.5237646,  0.09917503,  0.7036679 ],
    [0.5289718,  0.09073476,  0.70361465],
    [0.5343901,  0.08229896,  0.7035523 ],
    [0.54078937, 0.074453,    0.7034516 ],
    [0.5472114,  0.06677794,  0.7034085 ],
    [0.55377823, 0.05953822,  0.703696  ],
    [0.5603046,  0.05206416,  0.7037623 ],
    [0.56542146, 0.0433929,   0.70384246],
    [0.56787646, 0.03348375,  0.7038877 ],
    [0.5666649,  0.02346172,  0.7038103 ],
    [0.56244767, 0.0144706,   0.7038747 ],
    [0.5570772,  0.00599453,  0.7036372 ],
    [0.55133843, -0.00190075, 0.7034397 ],
    [0.5453437,  -0.01004345, 0.7034026 ],
    [0.53994703, -0.01828689, 0.7036807 ],
    [0.5348691,  -0.02686904, 0.7036231 ],
    [0.52967834, -0.03552297, 0.7035456 ],
    [0.52534956, -0.04476995, 0.70344186],
    [0.521081,   -0.05391448, 0.7033964 ],
    [0.5174823,  -0.06308798, 0.7036682 ],
    [0.5143985,  -0.07256885, 0.7036099 ],
    [0.51124674, -0.08214363, 0.70354515],
    [0.5089485,  -0.09205871, 0.70344836],
    [0.50657266, -0.10181558, 0.7034068 ],
    [0.5049495,  -0.11086131, 0.70362717],
    [0.50301147, -0.12068724, 0.7035563 ],
    [0.5019348,  -0.13076119, 0.70345193],
    [0.50081307, -0.14071679, 0.7034022 ],
    [0.50025153, -0.15049027, 0.70367306],
    [0.5000738,  -0.16041125, 0.7036151 ],
    [0.4996664,  -0.17045552, 0.7035423 ],
    [0.49989936, -0.18063611, 0.7034368 ],
    [0.4997488,  -0.19070452, 0.70339084],
    [0.499814,   -0.2005332,  0.70366883],
    [0.50002533, -0.21045949, 0.7036135 ],
    [0.49985456, -0.22016968, 0.70368856],
    [0.5001113,  -0.2279048,  0.70392925],
])

FINAL_SHAPE = np.array([[-0.00383,  0.04268,  0.00493],
 [ 0.00555,  0.04613,  0.00492],
 [ 0.01494,  0.04959,  0.00492],
 [ 0.02432,  0.05304,  0.00492],
 [ 0.03371,  0.05648,  0.00492],
 [ 0.04312,  0.05988,  0.00492],
 [ 0.05254,  0.06324,  0.00492],
 [ 0.06198,  0.06653,  0.00492],
 [ 0.07146,  0.06972,  0.00491],
 [ 0.08098,  0.07277,  0.00491],
 [ 0.08985,  0.07646,  0.00491],
 [ 0.09949,  0.07912,  0.00492],
 [ 0.11006,  0.07999,  0.00492],
 [ 0.12002,  0.08081,  0.00492],
 [ 0.13002,  0.08089,  0.00492],
 [ 0.14001,  0.08034,  0.00492],
 [ 0.14995,  0.07927,  0.00492],
 [ 0.15984,  0.07776,  0.00492],
 [ 0.16966,  0.07589,  0.00492],
 [ 0.17943,  0.07372,  0.00492],
 [ 0.18913,  0.07132,  0.00492],
 [ 0.1988 ,  0.06874,  0.00492],
 [ 0.20842,  0.06602,  0.00492],
 [ 0.21801,  0.06319,  0.00492],
 [ 0.22758,  0.06029,  0.00492],
 [ 0.23714,  0.05735,  0.00492],
 [ 0.24669,  0.05438,  0.00492],
 [ 0.25624,  0.05141,  0.00492],
 [ 0.26579,  0.04845,  0.00492],
 [ 0.27535,  0.04552,  0.00492],
 [ 0.28492,  0.04262,  0.00492],
 [ 0.29451,  0.03978,  0.00492],
 [ 0.30411,  0.03698,  0.00492],
 [ 0.31373,  0.03425,  0.00492],
 [ 0.32337,  0.03158,  0.00492],
 [ 0.33302,  0.02897,  0.00492],
 [ 0.34269,  0.02643,  0.00492],
 [ 0.35239,  0.02396,  0.00492],
 [ 0.36209,  0.02155,  0.00492],
 [ 0.37181,  0.01921,  0.00492],
 [ 0.38155,  0.01693,  0.00492],
 [ 0.3913 ,  0.01471,  0.00492],
 [ 0.40106,  0.01254,  0.00492],
 [ 0.41084,  0.01043,  0.00492],
 [ 0.42062,  0.00836,  0.00492],
 [ 0.43041,  0.00633,  0.00492],
 [ 0.44021,  0.00433,  0.00492],
 [ 0.45002,  0.00236,  0.00492],
 [ 0.45982,  0.00041,  0.00492],
 [ 0.46963, -0.00153,  0.00492],
 [ 0.47945, -0.00346,  0.00492]])


class ShapingSimplifiedEnv:
    def __init__(self, config: ShapingConfig):
        self.config = config
        gs.init(
            backend=gs.cpu if self.config.simulation.cpu else gs.gpu,
            logging_level="info",
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
        # Rotate TARGET_SHAPE by 90 degrees clockwise around the Z axis
        rotation_matrix = np.array([[0, 1, 0],
                        [-1, 0, 0],
                        [0, 0, 1]])
        rotated_target_shape = np.array(FINAL_SHAPE) @ rotation_matrix.T

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

        print(skeleton)

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
            path_period=1.0, # this is too fast, grasp is done better if 2.0 sec 
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
                self.obs_deque.append(obs)
                obs = np.stack(self.obs_deque)
                obs = obs.reshape(self.obs_horizon, -1)

                pred_action = self.model.run(obs)
                self.current_pred_action = pred_action
                self.ready_to_plot = True

                self.grasp(int(pred_action[0, 0]))
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



