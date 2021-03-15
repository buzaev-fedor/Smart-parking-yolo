from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet
from constants import *
from functions import *
import cv2
import time
from time import ctime
from flask import Flask, Response


print(cv2.__version__)
print(cv2.getBuildInformation())

"""Инициализация модели"""
m = Darknet(cfg_file)
m.print_network()
m.load_weights(weight_file)
print('Loading weights from %s... Done!'%(weight_file))
if use_cuda:
    m.cuda()

num_classes = m.num_classes
if num_classes == 20:
    names_file = 'data/voc.names'
elif num_classes == 80:
    names_file = 'data/coco.names'
else:
    names_file = 'data/x.names'

class_names = load_class_names(names_file)


"""Инициализая класса видеопотока"""
class ParkingDetector:
    # здесь подается ссылка на rtsp поток
    def __init__(self):
        video_source = ""
        rtsp_latency = 20
        g_stream = f"rtspsrc location={video_source} latency={rtsp_latency} ! decodebin ! videoconvert ! appsink"
        start = time.time()
        self.video = cv2.VideoCapture(g_stream, cv2.CAP_GSTREAMER)
        self.success, self.frame = self.video.read()
        """ здесь мы выбираем только часть фрейма и детектим только на этой части"""
        self.start_row = int(250)
        self.start_col = int(90)
        self.end_row = int(650)
        self.end_col = int(970)
        self.cropped_frame = self.frame[self.start_row:self.end_row, self.start_col:self.end_col]
        self.width, self.height = self.cropped_frame.shape[0], self.cropped_frame.shape[1]
        """Эти константы нужны, чтобы поток видео шел в real-time"""
        self.tracker = 6
        self.target = 6
        self.tmp = np.array([])
        self.time_list = []
        self.time_list.append(start)
        if not self.video.isOpened():
            print("Could not open feed")

    def __del__(self):
        self.video.release()

    """читаем видео, уменьшаем его и детектим. фун-я flow берет первый кадр, 
    отсчитывает итерации и берет следующий кадр
    """

    def get_frame(self):
        self.success, self.frame = self.video.read()
        while self.success:
            self.frame = cv2.resize(self.frame, (int(self.frame.shape[1]//2.4), int(self.frame.shape[0]//2.4)))
            self.cropped_frame = self.frame[self.start_row:self.end_row, self.start_col:self.end_col]
            return cv2.imencode('.jpg', self.flow())[1].tobytes()

    """
    функция, которая берет первый попавшийся кадр, прогоняет его через детекцию, а потом сохраняет его в озу и
    показывает дальше, когда переменная tracker итерируется
    """

    def flow(self):
        if self.tracker == self.target:
            height, width = self.cropped_frame.shape[0], self.cropped_frame.shape[1]
            start_time = time.time()
            self.cropped_frame = detect(self.cropped_frame, height, width)
            fps = 1/(time.time() - start_time)
            self.frame[self.start_row: self.start_row + height,
            self.start_col: self.start_col + width] = self.cropped_frame
            self.tmp = self.frame
            self.tracker = 0
            print(f"Time: {ctime(start_time)} FPS : {fps}")
            return self.frame
        else:
            self.tracker += 1
            return self.tmp


def detect(frame, height, width):
    overlay = frame.copy()
    """Переменные"""
    busy_car_places_id = np.array([])
    free_space = 0

    """Основной блок, тут детектятся машинки"""
    sized = cv2.resize(frame, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
    boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)

    """функция, где берется ббокс и координата середин машин"""
    car_box, center_car_box = car_boxes(boxes, height, width)

    """Рассчет по IoU для наших парковочных мест"""
    overlaps, bad_car_places_id = compute_overlaps(car_box, center_car_box)

    """Блок, где рисуется парковка"""
    try:
        for parking_id, overlap_areas in zip(dict_parked_car_places.keys(), overlaps):
            # max_IoU_overlap = np.max(overlap_areas)
            """ Тут идет сравнение с max_IoU_overlap """
            if np.max(overlap_areas) < 0.05:
                cv2.fillPoly(frame, [np.array(dict_parked_car_places[parking_id])], (120, 200, 132))
                free_space += 1
            if np.max(overlap_areas) >= 0.3:
                busy_car_places_id = np.append(busy_car_places_id, parking_id)
    except:
        pass

    """
    Блок, где выводятся плохие места
    Список не трогай!!!
    """
    busy_car_places_id = np.unique(busy_car_places_id)
    busy_car_places_id = (np.sort(busy_car_places_id)).astype(np.int32)
    busy_car_places_id_list = [number for number in busy_car_places_id]

    bad_car_places_id = np.unique(bad_car_places_id)
    bad_car_places_id = (np.sort(bad_car_places_id)).astype(np.int32)
    bad_car_places_id_list = [number for number in bad_car_places_id]

    """Вынужденная мера, чтобы отфильтровать список, нужна функция в функции"""
    def filter_duplicate(string_to_check):
        if string_to_check in busy_car_places_id_list:
            return False
        else:
            return True

    not_normal_car_places_id = list(filter(filter_duplicate, bad_car_places_id_list))
    """Здесь рисуется площадь хуевых тачек"""
    for place_id in not_normal_car_places_id:
        cv2.fillPoly(frame, [np.array(dict_parked_car_places[place_id])], (35, 51, 235))

    """Блок, где рисуются боксы и на вывод уходит"""
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    result_frame = plot_boxes_cv2(frame, boxes[0], savename=None, class_names=class_names)
    return np.array(result_frame)


app = Flask(__name__)


def gen(feed):
    while True:
        frame = feed.get_frame();
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(ParkingDetector()), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0')