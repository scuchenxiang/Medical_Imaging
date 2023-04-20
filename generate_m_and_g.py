import numpy as np
import generate_mi
import cv2
import matplotlib.pyplot as plt


set_theta1 = np.linspace(0, 6.28, 20)
set_theta2 = np.linspace(-1, 1, 512)
N_data = 20 * 512


def Generate_M(Set_theta1, Set_theta2):
    M = []
    index = 0
    for theta1 in Set_theta1:
        for theta2 in Set_theta2:
            Mi = generate_mi.Generate_Single_Mi(theta1, theta2, index)
            M.append(Mi)
            index += 1
    return M


def Generate_g(M, f):
    g = np.zeros((N_data, ))
    for i in range(N_data):
        for triple in M[i]:
            g[i] += triple[2] * f[triple[1]]
    return g

if __name__ == '__main__':
    M = Generate_M(set_theta1, set_theta2)
    img = cv2.imread('t.png', cv2.IMREAD_GRAYSCALE)
    resized_img = cv2.resize(img, (256, 256))
    f = resized_img.reshape((256 * 256,))

    g = Generate_g(M, f)
    plt.imshow(g.reshape((20 ,512)), cmap='gray')
    plt.show()