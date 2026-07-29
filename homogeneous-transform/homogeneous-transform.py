import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """
    # Your code here
    points = np.asarray(points)
    dim = points.ndim
    if dim == 1:
        points = points.reshape(1, -1)
    points = np.hstack([points, np.ones((points.shape[0], 1))])
    T = np.asarray(T)

    new_T = (T @ points.T).T
    # print(new_T)
    ans = new_T[:, :-1]
    # print(ans)
    return ans.tolist()[0] if dim == 1 else ans.tolist()

