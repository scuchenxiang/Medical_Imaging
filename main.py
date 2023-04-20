import numpy as np
import matplotlib.pyplot as plt
import cv2
import generate_m_and_g
import generate_mi
from TV_POCS import POCS_TV2


N_image = generate_mi.H * generate_mi.W
img = cv2.imread('t.png', cv2.IMREAD_GRAYSCALE)
resized_img = cv2.resize(img, (generate_mi.H, generate_mi.W))
f = resized_img.reshape((N_image,))


M = generate_m_and_g.Generate_M(generate_m_and_g.set_theta1, generate_m_and_g.set_theta2)
g = generate_m_and_g.Generate_g(M, f)


iters = 50

res = POCS_TV2(g, M, generate_m_and_g.N_data, N_image, generate_mi.H, generate_mi.W, iters, N_grad=20, a=0.2, eps=1e-8)
print(res.shape)

rec_img = res.reshape(generate_mi.H, generate_mi.W)
print(rec_img)
plt.imshow(rec_img, cmap='gray')
plt.show()
