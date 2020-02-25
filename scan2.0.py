import argparse
import cv2
import numpy as np
import imutils


def order_points(points):
    """"
    Функция, получающая на входе список точек, возвращает точки в упорядоченном виде
    :param points:  [верхний левый угол, верхний правый, нижний правый, нижний левый]
    """
    rect = np.zeros((4, 2), dtype="float32")

    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]

    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    return rect


def four_point_transform(_image, points):
    """
    Преобразование интересующей области изображения к виду "сверху"
    :param _image: исходное изображение
    :param points: точки прямоугольного контура интересующей области изображения
    :return: возвращает изображение с видом "сверху"
    """
    rect = order_points(points)
    (top_left, top_right, bottom_right, bottom_left) = rect

    width_a = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))
    width_b = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))
    min_width = min(int(width_a), int(width_b))

    height_a = np.sqrt(((top_right[0] - bottom_right[0]) ** 2) + ((top_right[1] - bottom_right[1]) ** 2))
    height_b = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2))
    min_height = min(int(height_a), int(height_b))

    transform_points = np.array([
        [0, 0],
        [min_width - 1, 0],
        [min_width - 1, min_height - 1],
        [0, min_height - 1]], dtype="float32")

    matrix = cv2.getPerspectiveTransform(rect, transform_points)
    _warped = cv2.warpPerspective(_image, matrix, (min_width, min_height))

    return _warped


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True)
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height=500)

blur_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
blur_image = cv2.GaussianBlur(blur_image, (3, 3), 0)

edged = cv2.Canny(blur_image, 40, 120)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

contours = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
total = 0
contours = imutils.grab_contours(contours)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

for point in contours:
    peri = cv2.arcLength(point, True)
    approx = cv2.approxPolyDP(point, 0.02 * peri, True)
    if len(approx) == 4:
        screen_cnt = approx
        break

cv2.drawContours(image, [screen_cnt], -1, (0, 255, 0), 2)
cv2.imwrite("contours3.jpg", image)

warped = four_point_transform(orig, screen_cnt.reshape(4, 2) * ratio)
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
cv2.imwrite("scanned3.jpg", warped)
print('Изображения сканированы')
