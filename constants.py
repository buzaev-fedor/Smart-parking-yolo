import os
from tool.torch_utils import *
import pickle5 as pickle

import shapely
from shapely.strtree import STRtree
from shapely.geometry import Point
from shapely.geometry import Polygon


"""Выдача и гиперпараметры"""
alpha = 0.6
use_cuda = True
ROOT_DIR = os.getcwd()
cfg_file = os.path.join(ROOT_DIR, "cfg/yolov4.cfg")
weight_file = os.path.join(ROOT_DIR, "weights/yolov4.weights")

"""Загрузка парковочных мест"""
regions = "regions.p"
with open(regions, 'rb') as f:
    parked_car_boxes = pickle.load(f)
    dict_parked_car_places = dict()
    for number in range(0, len(parked_car_boxes)):
        dict_parked_car_places.update({number: parked_car_boxes[number]})
"""
Расчет соседей парковочных мест и создание словаря, 
где ключ - это id парковочного места, 
а значение - это id соседних мест.
"""
dict_neighbors = dict()
for i in range(0, len(dict_parked_car_places.keys())):
    neighbors_list = []
    pol1_xy = Polygon(dict_parked_car_places[i])
    for j in range(0, len(dict_parked_car_places.keys())):
        pol2_xy = Polygon(dict_parked_car_places[j])
        polygon_intersection = pol1_xy.intersection(pol2_xy).area
        polygon_union = pol1_xy.union(pol2_xy).area
        IOU = polygon_intersection/polygon_union
        if IOU != 1:
            pol1_xy_bigger = shapely.affinity.scale(pol1_xy, xfact=2.0, yfact=2.0, origin='center')
            polygon_intersection = pol1_xy_bigger.intersection(pol2_xy).area
            polygon_union = pol1_xy_bigger.union(pol2_xy).area
            IOU = polygon_intersection/polygon_union
            if IOU != 0:
                neighbors_list.append(j)
    dict_neighbors.update({i: neighbors_list})

"""Рассчет псевдосередин у парковочных мест и создание Rtree"""
center_place_dict = {}
for number in dict_parked_car_places.keys():
    y1_place = dict_parked_car_places[number][0][0]
    y2_place = dict_parked_car_places[number][1][0]
    x1_place = dict_parked_car_places[number][0][1]
    x2_place = dict_parked_car_places[number][1][1]
    y1_middle = y1_place + (y2_place - y1_place)/2
    x1_middle = x1_place + (x2_place - x1_place)/2
    center_place = [y1_middle, x1_middle]
    center_place_dict.update({number: center_place})

"""Само создание RTree"""
points = [Point(center_place_dict[number][0], center_place_dict[number][1]) for number in center_place_dict.keys()]
tree_coord_place = STRtree(points)