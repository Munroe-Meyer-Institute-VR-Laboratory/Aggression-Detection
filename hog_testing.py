import cv2
import numpy as np
import math

from matplotlib import pyplot as plt


# def create_video_sample(video, image_size, tracklet, img_read=cv2.IMREAD_UNCHANGED):
#     frame_tensors = []
#     # writer = imageio.get_writer(f'debug_video.mp4',
#     #                             fps=8.0)
#     # height, width = 0, 0
#     # for still in self.accepted_c_stills:
#     #     if still.shape[0] > width:
#     #         width = still.shape[0]
#     #     if still.shape[1] > height:
#     #         height = still.shape[1]
#     # for still in self.accepted_c_stills:
#     #     writer.append_data(cv2.resize(still, (height, width)))
#     # writer.close()
#     for anno in video:
#         with open(anno) as f:
#             annotations = f.readlines()
#             annotation = None
#             if len(annotations) > 0:
#                 if len(annotations) == 1:
#                     annotation = annotations[0].strip()
#                 elif len([x for x in annotations if int(x.split(' ')[0]) == 0]) == 1:
#                     annotation = [x for x in annotations if int(x.split(' ')[0]) == 0][0]
#                 else:
#                     continue
#             if annotation and tracklet:
#                 if int(annotation.split(' ')[0]) == 0:
#                     _, x, y, w, h = map(float, annotation.split(' '))
#                     frame = cv2.imread(anno[:-3] + 'png', img_read)
#
#                     dh, dw = frame.shape[0], frame.shape[1]
#                     l = int((x - w / 2) * dw)
#                     t = int((y - h / 2) * dh)
#                     w = int(w * dw)
#                     h = int(h * dh)
#
#                     cropped_image = frame[t:t+h, l:l+w]
#                     resized_crop = cv2.resize(cropped_image, (image_size, image_size))
#                     # writer.append_data(resized_crop)
#                     if img_read == cv2.IMREAD_UNCHANGED:
#                         resized_frame = resized_crop[:, :, [0, 1, 2]]
#                     else:
#                         resized_frame = resized_crop
#                     frame_tensors.append(resized_frame)
#             else:
#                 # Just load the entire frame if no annotation found
#                 frame = cv2.imread(anno[:-3] + 'png', img_read)
#                 resized_crop = cv2.resize(frame, (image_size, image_size))
#                 # writer.append_data(resized_crop)
#                 resized_frame = resized_crop[:, :, [0, 1, 2]]
#                 frame_tensors.append(resized_frame)
#     return np.array(frame_tensors)


class HOG_Descriptor:
    def __init__(self, cell_size=16, bin_size=8):

        self.cell_size = cell_size
        self.bin_size = bin_size
        self.angle_unit = 360 / self.bin_size
        assert type(self.bin_size) == int, "bin_size should be integer,"
        assert type(self.cell_size) == int, "cell_size should be integer,"
        #assert type(self.angle_unit) == int, "bin_size should be divisible by 360"

    def predict(self, img, verbose):
        self.img = img
        self.img = np.sqrt(img / float(np.max(img)))
        self.img = self.img * 255
        height, width = self.img.shape
        gradient_magnitude, gradient_angle = self.global_gradient()
        gradient_magnitude = abs(gradient_magnitude)
        cell_gradient_vector = np.zeros((int(height / self.cell_size), int(width / self.cell_size), self.bin_size))
        for i in range(cell_gradient_vector.shape[0]):
            for j in range(cell_gradient_vector.shape[1]):
                cell_magnitude = gradient_magnitude[i * self.cell_size:(i + 1) * self.cell_size,
                                 j * self.cell_size:(j + 1) * self.cell_size]
                cell_angle = gradient_angle[i * self.cell_size:(i + 1) * self.cell_size,
                             j * self.cell_size:(j + 1) * self.cell_size]
                cell_gradient_vector[i][j] = self.cell_gradient(cell_magnitude, cell_angle)

        hog_image = self.render_gradient(np.zeros([height, width]), cell_gradient_vector)
        hog_vector = []
        for i in range(cell_gradient_vector.shape[0] - 1):
            for j in range(cell_gradient_vector.shape[1] - 1):
                block_vector = []
                block_vector.extend(cell_gradient_vector[i][j])
                block_vector.extend(cell_gradient_vector[i][j + 1])
                block_vector.extend(cell_gradient_vector[i + 1][j])
                block_vector.extend(cell_gradient_vector[i + 1][j + 1])
                mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
                magnitude = mag(block_vector)
                if magnitude != 0:
                    normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
                    block_vector = normalize(block_vector, magnitude)
                hog_vector.append(block_vector)
        hog_vector = np.array([item for sublist in hog_vector for item in sublist])
        return hog_vector #, hog_image

    def global_gradient(self):
        gradient_values_x = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=5)
        gradient_values_y = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0)
        gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)
        return gradient_magnitude, gradient_angle

    def cell_gradient(self, cell_magnitude, cell_angle):
        orientation_centers = [0] * self.bin_size
        for i in range(cell_magnitude.shape[0]):
            for j in range(cell_magnitude.shape[1]):
                gradient_strength = cell_magnitude[i][j]
                gradient_angle = cell_angle[i][j]
                min_angle, max_angle, mod = self.get_closest_bins(gradient_angle)
                orientation_centers[min_angle] += (gradient_strength * (1 - (mod / self.angle_unit)))
                orientation_centers[max_angle] += (gradient_strength * (mod / self.angle_unit))
        return orientation_centers

    def get_closest_bins(self, gradient_angle):
        idx = int(gradient_angle / self.angle_unit)
        mod = gradient_angle % self.angle_unit
        if idx == self.bin_size:
            return idx - 1, (idx) % self.bin_size, mod
        return idx, (idx + 1) % self.bin_size, mod

    def render_gradient(self, image, cell_gradient):
        cell_width = self.cell_size / 2
        max_mag = np.array(cell_gradient).max()
        for x in range(cell_gradient.shape[0]):
            for y in range(cell_gradient.shape[1]):
                cell_grad = cell_gradient[x][y]
                cell_grad /= max_mag
                angle = 0
                angle_gap = self.angle_unit
                for magnitude in cell_grad:
                    angle_radian = math.radians(angle)
                    x1 = int(x * self.cell_size + magnitude * cell_width * math.cos(angle_radian))
                    y1 = int(y * self.cell_size + magnitude * cell_width * math.sin(angle_radian))
                    x2 = int(x * self.cell_size - magnitude * cell_width * math.cos(angle_radian))
                    y2 = int(y * self.cell_size - magnitude * cell_width * math.sin(angle_radian))
                    cv2.line(image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
                    angle += angle_gap
        return image

#
# tracklet = create_video_sample([r"C:\GitHub\Keypoint-LSTM\datasets\stills\MMISB Cropped\p001\biting\0\0.txt"], 256, True, img_read=cv2.IMREAD_GRAYSCALE)
# # img = cv2.imread(r"C:\GitHub\Keypoint-LSTM\datasets\stills\MMISB Cropped\p001\biting\0\0.png", cv2.IMREAD_GRAYSCALE)
# hog = HOG_Descriptor(cell_size=8, bin_size=8)
# vector, image = hog.predict(tracklet[0], 0)
# plt.imshow(np.hstack((image, tracklet[0])), cmap='gray')
# plt.show()