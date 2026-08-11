import numpy as np

def rotate_around_z(points, theta):
    """
    Rotate 3D point(s) around the Z-axis by angle theta (radians).
    """
    rot_mat = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1],
    ])

    np_points = np.asarray(points)
    points_dim = np_points.ndim

    if points_dim == 1:
        np_points = np_points.reshape(1, -1)

    rot_points = np_points @ rot_mat.T

    if points_dim == 1:
        rot_points = np.squeeze(rot_points)

    return rot_points
