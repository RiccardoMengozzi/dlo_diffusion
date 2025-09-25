import numpy as np


def normalize_action(dlo_0, a, cs0, csR, disp_scale, angle_scale, scale_action=True, rot_check_flag=None):
    a1 = normalize_action_idx(a[0], num_points=dlo_0.shape[0], rot_check_flag=rot_check_flag)

    # compute the target position of the active edge based on the initial shape and the action
    points_grasp, points_place = compute_edges_points_from_action(dlo_0, a)

    points_grasp_up = np.dot(csR, (points_grasp - cs0).T).T
    points_place_up = np.dot(csR, (points_place - cs0).T).T

    action_new = compute_action_from_edges_points(points_grasp_up, points_place_up)

    # scale values if necessary
    a23 = action_new[:2] / np.array([disp_scale, disp_scale]) if scale_action else action_new[:2]
    a4 = action_new[-1] / angle_scale if scale_action else action_new[-1]

    return np.stack([a1, a23[0], a23[1], a4], axis=-1)


def compute_edges_points_from_action(init_pos, action):
    idx = int(action[0])
    dtheta = action[3]

    node_0_pos = init_pos[idx, :]
    node_1_pos = init_pos[idx + 1, :]

    edge_pos = (node_1_pos + node_0_pos) / 2
    edge_dir = node_1_pos - node_0_pos
    edge_len = np.linalg.norm(edge_dir)
    edge_dir = edge_dir / edge_len

    # new pos
    new_edge_pos_x = edge_pos[0] + action[1]
    new_edge_pos_y = edge_pos[1] + action[2]
    new_edge_pos = np.array([new_edge_pos_x, new_edge_pos_y])

    # new dir
    new_edge_x = edge_dir[0] * np.cos(dtheta) - edge_dir[1] * np.sin(dtheta)
    new_edge_y = edge_dir[0] * np.sin(dtheta) + edge_dir[1] * np.cos(dtheta)
    new_edge_dir = np.array([new_edge_x, new_edge_y])

    pos0_tgt = new_edge_pos - new_edge_dir * edge_len / 2
    pos1_tgt = new_edge_pos + new_edge_dir * edge_len / 2

    points_grasp = np.array([node_0_pos, node_1_pos]).squeeze()
    points_place = np.array([pos0_tgt, pos1_tgt]).squeeze()
    return points_grasp, points_place


def compute_action_from_edges_points(points_grasp, points_place):
    center_grasp_up = points_grasp.mean(axis=0, keepdims=True)
    center_place_up = points_place.mean(axis=0, keepdims=True)

    dir_grasp_up = points_grasp[1, :] - points_grasp[0, :]
    dir_grasp_up = dir_grasp_up / np.linalg.norm(dir_grasp_up)

    dir_place_up = points_place[1, :] - points_place[0, :]
    dir_place_up = dir_place_up / np.linalg.norm(dir_place_up)

    angle = np.arctan2(dir_place_up[1], dir_place_up[0]) - np.arctan2(dir_grasp_up[1], dir_grasp_up[0])
    disp_x = center_place_up[0, 0] - center_grasp_up[0, 0]
    disp_y = center_place_up[0, 1] - center_grasp_up[0, 1]
    return np.array([disp_x, disp_y, angle])


def normalize_action_idx(idx, num_points, rot_check_flag):
    if rot_check_flag is None:
        raise ValueError("rot_check_flag is None")

    if rot_check_flag:
        idx = (num_points - 2) - idx  # reverse the index of the active edge

    return idx / (num_points - 1.0)


def compute_cs0_csR(dlo_0):
    cs0 = np.mean(dlo_0, axis=0, keepdims=True)
    dlo_0_centred = dlo_0 - cs0
    cov = dlo_0_centred.T @ dlo_0_centred
    eigval, eigvec = np.linalg.eig(cov)
    csR = eigvec[:, np.argsort(eigval)[::-1]].T
    return cs0, csR


def normalize_dlo(dlo, cs0, csR):
    return np.dot(csR, (dlo - cs0).T).T


def check_rot_and_flip(dlo_0_n, dlo_1_n):
    # flipping the dlo of the wrong order and adjusting the action correspondingly
    if dlo_0_n[0, 0] > 0.0:
        return dlo_0_n[::-1], dlo_1_n[::-1], True

    return dlo_0_n, dlo_1_n, False


def normalize(dlo_0, dlo_1, action_traj, disp_scale, angle_scale):
    # compute normalization factors
    cs0, csR = compute_cs0_csR(dlo_0)

    # dlo shape
    dlo_0_n = normalize_dlo(dlo_0, cs0, csR)
    dlo_1_n = normalize_dlo(dlo_1, cs0, csR)

    # check rot
    dlo_0_n, dlo_1_n, rot_check_flag = check_rot_and_flip(dlo_0_n, dlo_1_n)

    # action
    action_traj_n = []
    for action in action_traj:
        action_n = normalize_action(
            dlo_0, action, cs0, csR, disp_scale, angle_scale, scale_action=True, rot_check_flag=rot_check_flag
        )
        action_traj_n.append(action_n)
    action_traj_n = np.array(action_traj_n)

    return dlo_0_n, dlo_1_n, action_traj_n, cs0, csR, rot_check_flag


###############################


def denormalize_action_idx(idx, num_points, rot_check_flag):
    idx = idx * (num_points - 1.0)

    if rot_check_flag:
        idx = (num_points - 2) - idx  # reverse the index of the active edge

    return idx


def denormalize_dlo(dlo, cs0, csR, rot_check_flag=False):
    if rot_check_flag:
        dlo = dlo[::-1]
    return (csR.T @ dlo.T).T + cs0


def denormalize_action(dlo_0_n, a, cs0, csR, disp_scale, angle_scale, rot_check_flag=False):
    a1 = denormalize_action_idx(a[0], num_points=dlo_0_n.shape[0], rot_check_flag=rot_check_flag)

    # a1 between 0 and dlo_0_n.shape[0] - 2
    if a1 < 0:
        a1 = 0
    elif a1 > dlo_0_n.shape[0] - 2:
        a1 = dlo_0_n.shape[0] - 2

    a23 = a[1:3]
    a4 = a[3]

    a23 = a23 * np.array([disp_scale, disp_scale])
    a4 = a4 * angle_scale

    a_new = np.array([a1, a23[0], a23[1], a4])

    # compute the target position of the active edge based on the initial shape and the action
    points_grasp, points_place = compute_edges_points_from_action(dlo_0_n, a_new)

    points_grasp_up = np.dot(csR.T, points_grasp.T).T + cs0
    points_place_up = np.dot(csR.T, points_place.T).T + cs0

    action_new = compute_action_from_edges_points(points_grasp_up, points_place_up)

    return np.stack([a1, action_new[0], action_new[1], action_new[2]], axis=-1)


def denormalize(dlo_0_n, dlo_1_n, action_n, cs0, csR, disp_scale, angle_scale, rot_check_flag=False):
    dlo_0 = denormalize_dlo(dlo_0_n, cs0, csR, rot_check_flag)
    dlo_1 = denormalize_dlo(dlo_1_n, cs0, csR, rot_check_flag)
    action = denormalize_action(dlo_0_n, action_n, cs0, csR, disp_scale, angle_scale, rot_check_flag)

    return dlo_0, dlo_1, action


def denormalize_action_horizon(dlo_0, action_n, cs0, csR, disp_scale, angle_scale, rot_check_flag=False):
    h = action_n.shape[0]
    actions = []
    for i in range(h):
        action = denormalize_action(dlo_0, action_n[i], cs0, csR, disp_scale, angle_scale, rot_check_flag)
        actions.append(action)

    return np.stack(actions, axis=0)


def denormalize_horizon(dlo_0_n, dlo_1_n, action_n, cs0, csR, disp_scale, angle_scale, rot_check_flag=False):

    dlo_0 = denormalize_dlo(dlo_0_n, cs0, csR, rot_check_flag)
    dlo_1 = denormalize_dlo(dlo_1_n, cs0, csR, rot_check_flag)

    actions = denormalize_action_horizon(dlo_0, action_n, cs0, csR, disp_scale, angle_scale, rot_check_flag)

    return dlo_0, dlo_1, actions


def convert_action_horizon_to_absolute(dlo_state, action_horizon):

    idx_action = int(action_horizon[:, 0][0])

    point_0 = dlo_state[idx_action]
    point_1 = dlo_state[idx_action + 1]
    angle = np.arctan2(point_1[1] - point_0[1], point_1[0] - point_0[0])

    list_action_points = []
    list_action_points.append(point_0)

    list_action_rot = []
    list_action_rot.append(angle)

    action_horizon_13 = action_horizon[:, 1:]

    for i in range(1, action_horizon.shape[0]):
        action_point = list_action_points[-1] + action_horizon_13[i, :2]
        list_action_points.append(action_point)

        action_rot = list_action_rot[-1] + action_horizon_13[i, 2]
        list_action_rot.append(action_rot)

    return np.stack(list_action_points, axis=0), np.stack(list_action_rot, axis=0)