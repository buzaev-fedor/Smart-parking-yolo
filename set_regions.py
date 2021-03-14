import os
import numpy as np
import cv2
import pickle
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.widgets import PolygonSelector
from matplotlib.collections import PatchCollection
from shapely.geometry import box
from shapely.geometry import Polygon as shapely_poly


points = []
prev_points = []
patches = []
total_points = []
breaker = False


class SelectFromCollection(object):
    def __init__(self, ax):
        self.canvas = ax.figure.canvas

        self.poly = PolygonSelector(ax, self.onselect)
        self.ind = []

    def onselect(self, verts):
        global points
        points = verts
        self.canvas.draw_idle()

    def disconnect(self):
        self.poly.disconnect_events()
        self.canvas.draw_idle()


def break_loop(event):
    global breaker
    global globSelect
    global savePath
    if event.key == 'b':
        globSelect.disconnect()
        if os.path.exists(savePath):
            os.remove(savePath)

        print("data saved in "+ savePath + " file")    
        with open(savePath, 'wb') as f:
            pickle.dump(total_points, f, protocol=pickle.HIGHEST_PROTOCOL)
            exit()


def onkeypress(event):
    global points, prev_points, total_points
    if event.key == 'n': 
        pts = np.array(points, dtype=np.int32)   
        if points != prev_points and len(set(points)) == 4:
            print("Points : "+str(pts))
            patches.append(Polygon(pts))
            total_points.append(pts)
            prev_points = points


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', help="Path of video file")
    parser.add_argument('--out_file', help="Name of the output file", default="regions.p")
    args = parser.parse_args()

    global globSelect
    global savePath
    savePath = args.out_file if args.out_file.endswith(".p") else args.out_file+".p"

    print("\n> Выберите область на рисунке, нарисовав ее четырехугольником.")
    print("> Нажмите 'f' чтобы развернуть на полный экран.")
    print("> Нажмите 'esc' для того не сохранять последнюю область.")
    print("> Зажмите 'shift' для того, чтобы перенести все контуры четырехугольников.")
    print("> Зажмите 'ctrl' чтобы перенести вершину четырехугольника.")
    print("> После создания четырехугольника, нажмите 'n' чтобы сохоанить текущий четырехугольник и нажмите 'q' чтобы "
          "нарисовать новый")
    print("> Когда вы закончите, нажмите 'b' чтобы выйти из программы\n")
    
    video_capture = cv2.VideoCapture(args.video_path)

    cnt=0
    rgb_image = None
    while video_capture.isOpened():
        success, frame = video_capture.read()
        start_row = int(250)
        start_col = int(90)
        end_row = int(650)
        end_col = int(970)
        # frame = frame[start_row:end_row, start_col:end_col]
        if not success:
            break
        if cnt == 5:
            rgb_image = frame[start_row:end_row, start_col:end_col, ::-1]
        cnt += 1
    video_capture.release()

    while True:
        fig, ax = plt.subplots()
        image = rgb_image
        ax.imshow(image)
    
        p = PatchCollection(patches, alpha=0.7)
        p.set_array(10*np.ones(len(patches)))
        ax.add_collection(p)
            
        globSelect = SelectFromCollection(ax)
        bbox = plt.connect('key_press_event', onkeypress)
        break_event = plt.connect('key_press_event', break_loop)
        plt.show()
        globSelect.disconnect()