import numpy as np
import cv2
from constants import *

"""функция, где берется ббокс и координата середин машин"""
def car_boxes(boxes, height, width):
    car_box = []
    center_car_box = []
    for car_id in range(len(boxes[0])):
        box = boxes[0][car_id]
        y1 = int(box[0]*width)
        x1 = int(box[1]*height)
        y2 = int(box[2]*width)
        x2 = int(box[3]*height)
        p1 = (y1, x1)
        p2 = (y2, x1)
        p3 = (y2, x2)
        p4 = (y1, x2)
        center_car_box.append([(y1 + (y2 - y1)/2), (x1 + (x2 - x1)/2)])
        car_box.append((p3, p4, p1, p2))

    return car_box, center_car_box


"""Вспомогательная функция, где мы вычисляем по метрике IoU парковочные места"""
def compute_overlaps(car_box, center_car_box):
    overlaps = np.zeros((len(dict_parked_car_places), len(car_box)))
    bad_car_places_id = np.array([])
    """В следующей строке я нахожу центральную точку ббокса и увеличиваю ее в 265 раз(подобранный элемент)"""
    for num in range(len(car_box)):
        """Выделяется полигон машины"""
        polygon_2 = Polygon(car_box[num])
        """Здесь я уменьшаю координаты объекта по x и y"""
        polygon_2 = shapely.affinity.scale(polygon_2, xfact=0.6, yfact=0.5, origin='center')
        """Нахожу координату центра ббокса и ищу ближайших соседей"""
        car_box_point = Point(center_car_box[num][0], center_car_box[num][1]).buffer(100)
        nearest_coord = ([o.wkt for o in tree_coord_place.query(car_box_point) if o.intersects(car_box_point)])
        coord_list = []
        for point in nearest_coord:
            point = [float(point.strip('()')) for point in point.split()[1:]]
            coord_list.append(point)
        nearest_ids = []
        for item in coord_list:
            result = [key for key, value in center_place_dict.items() if value == item]
            nearest_ids.append(result)
        for nearest_id in nearest_ids:
            nearest_id = nearest_id[0]
            neighbors = dict_neighbors[nearest_id]
            polygon_1 = Polygon(dict_parked_car_places[nearest_id])
            # intersection = polygon_1.intersection(polygon_2).area
            # union = polygon_1.union(polygon_2).area
            # IOU = intersection/union
            IOU = polygon_1.intersection(polygon_2).area/polygon_1.union(polygon_2).area
            if IOU == 0:
                overlaps[nearest_id][num] = IOU
            else:
                """Здесь идет расчет для соседних мест"""
                for neighbor in neighbors:
                    polygon_neighbor = Polygon(dict_parked_car_places[neighbor])
                    # intersection_neighbor = polygon_neighbor.intersection(polygon_2).area
                    # union_neighbor = polygon_neighbor.intersection(polygon_2).area
                    try:
                        # IOU_neighbor = intersection_neighbor/union_neighbor
                        IOU_neighbor = polygon_neighbor.intersection(polygon_2).area/polygon_neighbor.intersection(
                            polygon_2).area
                        if IOU_neighbor != 0:
                            bad_car_places_id = np.append(bad_car_places_id, neighbor)
                            overlaps[nearest_id][num] = IOU
                    except:
                        overlaps[nearest_id][num] = IOU

    return np.array(overlaps), bad_car_places_id
