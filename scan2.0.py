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

    # верхний левый угол будет иметь наименьшую сумму координат, нижний правый - наибольшую
    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]

    # верхний правый угол будет иметь наименьшую разницу координат, нижний левый - наибольшую
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

    # определяем ширину нового изображения, как разницу между верхними или нижними точками по оси X
    # выбираем наименьшее расстояние для снижения эффекта растягивания
    width_a = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))
    width_b = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))
    min_width = min(int(width_a), int(width_b))

    # проделывает аналогичную операцию для высоты по оси Y
    height_a = np.sqrt(((top_right[0] - bottom_right[0]) ** 2) + ((top_right[1] - bottom_right[1]) ** 2))
    height_b = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2))
    min_height = min(int(height_a), int(height_b))

    # преобразуем полученные координаты к массиву [верхний левый угол[x, y],
    # верхний правый[x, y], нижний правый[x, y], нижний левый[x, y]]
    transform_points = np.array([
        [0, 0],
        [min_width - 1, 0],
        [min_width - 1, min_height - 1],
        [0, min_height - 1]], dtype="float32")

    # преобразование матрицы
    matrix = cv2.getPerspectiveTransform(rect, transform_points)
    _warped = cv2.warpPerspective(_image, matrix, (min_width, min_height))

    # возвращаем преобразованное изображение
    return _warped


# парсим входные аргументы
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image to be scanned")
args = vars(ap.parse_args())

# загружаем изображение и изменяем размеры для более точного определения контуров документа
image = cv2.imread(args["image"])
# сохранем пропорции исходного изобрадения
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height=500)

#  конвертируем изображение в цветовую схему YUV для лучшей работы с белым цветом
#  и "размываем" изображения для устранения высокочастотных шумов
blur_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
blur_image = cv2.GaussianBlur(blur_image, (3, 3), 0)

# определяем контуры
edged = cv2.Canny(blur_image, 40, 120)

# для надежности применим оперцию закрытия контуров
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

# вычисляем контуры, сохраняя самые большие
contours = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
total = 0
contours = imutils.grab_contours(contours)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# создаем цикл по контуру, аппроксимируем точки
for point in contours:
    peri = cv2.arcLength(point, True)
    approx = cv2.approxPolyDP(point, 0.02 * peri, True)
    if len(approx) == 4:
        screen_cnt = approx
        break

#  Наносим контуры на исходное изобрадение и сохраняем его
cv2.drawContours(image, [screen_cnt], -1, (0, 255, 0), 2)
cv2.imwrite("contours3.jpg", image)

# применяем функцию к изображению для создания эффекта "вид сверху", матрицу координат умножаем на коэффициент
# пропорции для восстановления исходных размеров, возвращаемся к стандартной цветовой схеме и сохраняем изображение
warped = four_point_transform(orig, screen_cnt.reshape(4, 2) * ratio)
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
cv2.imwrite("scanned3.jpg", warped)
print('Изображения сканированы')
