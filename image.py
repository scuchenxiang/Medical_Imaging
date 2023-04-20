import cv2


img = cv2.imread('t.png', cv2.IMREAD_GRAYSCALE)
resized_img = cv2.resize(img, (256, 256))

print(resized_img)

cv2.imshow("resized_img", resized_img)
cv2.waitKey(0)
