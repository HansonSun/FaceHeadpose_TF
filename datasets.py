import os
import numpy as np
import cv2
import pandas as pd
from PIL import Image, ImageFilter
import utils

def get_list_from_filenames(file_path):
    # input:    relative path to .txt file with file names
    # output:   list of relative path names
    with open(file_path) as f:
        lines = f.read().splitlines()
    return lines

class Pose_300W_LP(object):
    # Head pose from 300W-LP dataset
    def __init__(self, data_dir="", filename_path="./label/300W_LP_PATH.txt", img_ext='.jpg', annot_ext='.mat', image_mode='RGB'):
        self.data_dir = data_dir
        self.img_ext = img_ext
        self.annot_ext = annot_ext

        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    def generate(self):
        for index in range(self.length):

            img = Image.open(os.path.join(self.data_dir, self.X_train[index] + self.img_ext))
            img = img.convert(self.image_mode)
            mat_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)

            # Crop the face loosely
            pt2d = utils.get_pt2d_from_mat(mat_path)
            x_min = min(pt2d[0,:])
            y_min = min(pt2d[1,:])
            x_max = max(pt2d[0,:])
            y_max = max(pt2d[1,:])

            # k = 0.2 to 0.40
            k = np.random.random_sample() * 0.2 + 0.2
            x_min -= 0.6 * k * abs(x_max - x_min)
            y_min -= 2 * k * abs(y_max - y_min)
            x_max += 0.6 * k * abs(x_max - x_min)
            y_max += 0.6 * k * abs(y_max - y_min)
            img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

            # We get the pose in radians
            pose = utils.get_ypr_from_mat(mat_path)
            # And convert to degrees.
            pitch = pose[0] * 180 / np.pi
            yaw   = pose[1] * 180 / np.pi
            roll  = pose[2] * 180 / np.pi

            # Bin values
            bins = np.array(range(-99, 99, 3))
            binned_pose = np.digitize([yaw, pitch, roll], bins)

            # Get target tensors
            labels = binned_pose
            cont_labels = np.array([yaw, pitch, roll])

            yield img, labels, cont_labels, self.X_train[index]



if __name__ == "__main__":
    testdataset = Pose_300W_LP(None)
    print(testdataset.length)
    for img, labels, cont_labels, imgpath in testdataset.generate():
        nimg=np.array(img)
        # print(type(labels))
        # print ("yaw",cont_labels[0],"pitch",cont_labels[1],"row",cont_labels[2])
        nimg = cv2.resize(nimg,(400,400))
        # utils.plot_pose_cube(nimg, cont_labels[0], cont_labels[1], cont_labels[2], tdx=None, tdy=None, size=150.)
        utils.draw_axis(nimg, cont_labels[0], cont_labels[1], cont_labels[2], tdx=200, tdy=200, size = 100)
        cv2.imshow("e",nimg)
        cv2.waitKey(0) 