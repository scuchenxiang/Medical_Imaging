import numpy as np

H_length = 20
W_length = 20
H = 256
W = 256
dx = H_length / H
dy = H_length / W
Y_plane_start = - (W_length / 2)
Y_plane_end = W_length / 2
X_plane_start = - (H_length / 2)
X_plane_end = H_length / 2
alpha_x = np.zeros((H + 2,))
alpha_y = np.zeros((W + 2,))


def Generate_Single_Mi(theta1, theta2, index):
    """
    inputs:
    theta1:源点与旋转中心的连线 与 Y轴 的夹角
    theta2:射线 与 源点与旋转中心的连线 的夹角
    index:the index of ray

    outputs:
    ret: 2-d list, sparse representation of one row of the system matrix
    """
    X1 = -40 * np.sin(theta1)
    Y1 = -40 * np.cos(theta1)
    X2 = 40 * np.sin(theta1) - 80 * np.tan(theta2) * np.cos(theta1)
    Y2 = 40 * np.cos(theta1) + 80 * np.tan(theta2) * np.sin(theta1)

    # compute sides
    alpha_x[1] = (X_plane_start - X1) / (X2 - X1)
    alpha_x[H + 1] = (X_plane_end - X1) / (X2 - X1)
    alpha_x[1: H + 2] = np.linspace(alpha_x[1], alpha_x[H + 1], H + 1)

    alpha_y[1] = (Y_plane_start - Y1) / (Y2 - Y1)
    alpha_y[H + 1] = (Y_plane_end - Y1) / (Y2 - Y1)
    alpha_y[1: H + 2] = np.linspace(alpha_y[1], alpha_y[H + 1], H + 1)

    # calculate range of parametric values
    alpha_min = np.max((0, np.minimum(alpha_x[1], alpha_x[H + 1]), np.minimum(alpha_y[1], alpha_y[H + 1])))
    alpha_max = np.min((1, np.maximum(alpha_x[1], alpha_x[H + 1]), np.maximum(alpha_y[1], alpha_y[H + 1])))

    if alpha_min >= alpha_max:
        return []

    # calculate range of indices
    if X2 - X1 > 0:
        i_min = int(H + 1 - np.floor((X_plane_end - alpha_min * (X2 - X1) - X1) / dx))
        i_max = int(1 + np.floor((X1 + alpha_max * (X2 - X1) - X_plane_start) / dx))
    if X2 - X1 < 0:
        i_min = int(H + 1 - np.floor((X_plane_end - alpha_max * (X2 - X1) - X1) / dx))
        i_max = int(1 + np.floor((X1 + alpha_min * (X2 - X1) - X_plane_start) / dx))

    if Y2 - Y1 > 0:
        j_min = int(H + 1 - np.floor((Y_plane_end - alpha_min * (Y2 - Y1) - Y1) / dy))
        j_max = int(1 + np.floor((Y1 + alpha_max * (Y2 - Y1) - Y_plane_start) / dy))
    if Y2 - Y1 < 0:
        j_min = int(H + 1 - np.floor((Y_plane_end - alpha_max * (Y2 - Y1) - Y1) / dy))
        j_max = int(1 + np.floor((Y1 + alpha_min * (Y2 - Y1) - Y_plane_start) / dy))

    # calculate parametric sets
    set_x = alpha_x[i_min: i_max + 1]
    set_y = alpha_y[j_min: j_max + 1]

    # merge sets to form set_alpha
    set_alpha = np.concatenate((set_x, set_y, np.array([alpha_min, alpha_max])))
    set_alpha = np.sort(set_alpha)

    n = len(set_alpha) - 1
    d12 = np.sqrt((X2 - X1) * (X2 - X1) + (Y2 - Y1) * (Y2 - Y1))

    # calculate voxel lengths and indices
    Mi = []
    for m in range(1, n + 1):
        lm = d12 * (set_alpha[m] - set_alpha[m - 1])
        if lm != 0:
            alpha_mid = (set_alpha[m] + set_alpha[m - 1]) / 2
            i = 1 + (X1 + alpha_mid * (X2 - X1) - X_plane_start) // dx
            j = 1 + (Y1 + alpha_mid * (Y2 - Y1) - Y_plane_start) // dy
            k = int((i - 1) * W + (j - 1))
            Mi.append([index, k, lm])

    return Mi


if __name__ == '__main__':
    Mi = Generate_Single_Mi(0, 0.2, 0)
    print(Mi)
