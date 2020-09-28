import cv2
import os
# Please modify the path
path="DRIVE/train/images"
save="Drive/flip/images/"
for name in os.listdir(path):
    image = cv2.imread(path+name)

    # Flipped Horizontally
    h_flip = cv2.flip(image, 1)
    cv2.imwrite(save+"h"+name, h_flip)



    # Flipped Vertically
    v_flip = cv2.flip(image, 0)
    cv2.imwrite(save+"v"+name, v_flip)

    # Flipped Horizontally & Vertically
    hv_flip = cv2.flip(image, -1)
    cv2.imwrite(save+"hv"+name, hv_flip)
