import numpy as np


def POCS_TV2(g, M, N_data, N_image, H, W, iters, N_grad=20, a=0.2, eps=1e-8):
    """
    input:
    g: projection-data with shape (N_data, )
    M: System Matrix with shape (N_data, N_image)
    N_image : shape of (H, W)
    iters : number of iterations

    output:
    f_TV_POS[iters - 1] : reconstructed image
    """
    # Initialization
    f_TV_DATA = np.zeros((iters + 1, N_image))
    f_TV_POS = np.zeros((iters, N_image))
    f_TV_GRAD = np.zeros((iters, N_image))
    v_st = np.zeros((iters, H, W))
    v = np.zeros((iters, N_image))
    v_std = np.zeros((iters, N_image))

    for n in range(iters):
        # data projection iteration
        for m in range(1, N_data):
            tmp = tmp2 = 0
            M_full_m = np.zeros((N_image,))
            for triple in M[m]:
                tmp += triple[2] * f_TV_DATA[n][triple[1]]
                tmp2 += triple[2] * triple[2]
                M_full_m[triple[1]] = triple[2]

            f_TV_DATA[n] = f_TV_DATA[n] + M_full_m * (g[m] - tmp) / (tmp2 + 1e-5)

        # Positivity constraint
        f_TV_POS[n] = f_TV_DATA[n] * (f_TV_DATA[n] >= 0)

        # TV gradient descent initialization
        f_TV_GRAD[n] = f_TV_POS[n]
        d_A = np.sqrt(np.dot(f_TV_DATA[n] - f_TV_POS[n], f_TV_DATA[n] - f_TV_POS[n]))

        # TV gradient descent
        for m in range(1, N_grad):
            f = f_TV_GRAD[n].reshape(H, W)
            for s in range(1, H - 1):
                for t in range(1, W - 1):
                    # compute TV gradient
                    v_st[n, s, t] = (((f[s, t] - f[s - 1, t]) + (f[s, t] - f[s, t - 1])) / np.sqrt(
                        eps + (f[s, t] - f[s - 1, t]) * (f[s, t] - f[s - 1, t]) + (f[s, t] - f[s, t - 1]) * (
                                    f[s, t] - f[s, t - 1]))
                                     - (f[s + 1, t] - f[s, t]) / np.sqrt(
                                eps + (f[s + 1, t] - f[s, t]) * (f[s + 1, t] - f[s, t]) + (
                                            f[s + 1, t] - f[s + 1, t - 1]) * (f[s + 1, t] - f[s + 1, t - 1]))
                                     - (f[s, t + 1] - f[s, t]) / np.sqrt(
                                eps + (f[s, t + 1] - f[s, t]) * (f[s, t + 1] - f[s, t]) + (
                                            f[s, t + 1] - f[s - 1, t + 1]) * (f[s, t + 1] - f[s - 1, t + 1])))

            v[n] = v_st[n].reshape(N_image, )
            v_std[n] = v[n] / np.sqrt(np.dot(v[n], v[n]))
            f_TV_GRAD[n] = f_TV_GRAD[n] - a * d_A * v_std[n]

        # Initialize next loop
        f_TV_DATA[n + 1] = f_TV_GRAD[n]

    return f_TV_POS[iters - 1]
