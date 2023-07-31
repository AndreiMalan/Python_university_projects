import cv2
import numpy as np
from google.colab.patches import cv2_imshow
from google.colab import files
image=cv2.imread("poza1.jpeg")
gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray=cv2.medianBlur(gray, 5)
edges=cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
color=cv2.bilateralFilter(image, 9, 250, 250)
cartoon=cv2.bitwise_and(color, color, mask=edges)
cartoon = cv2.stylization(image, sigma_s=500, sigma_r=0.35)

#cv2_imshow(image)
cv2_imshow(edges)
cv2_imshow(cartoon)
cv2.imwrite("cartoon_image1.jpg", cartoon)
cv2.imwrite("edge_image1.jpg", edges)
files.download("cartoon_image1.jpg")
files.download("edge_image1.jpg")