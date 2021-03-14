import sys
import os
import time
import argparse
from collections import namedtuple
import numpy as np
import cv2
# from PIL import Image
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

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

"""Инициализая класса видеопотока"""
class ParkingDetector:
    # здесь подается ссылка на rtsp поток
    def __init__(self):
        video_source = "rtsp://test:jrqoDUAU5o@194.186.3.122:7792"
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
        self.tracker = 6
        self.target = 6
        self.tmp = np.array([])
        self.time_list = []
        self.time_list.append(start)
        # TensorRT constant
        self.engine = get_engine("yolov4_1_3_608_608.engine")
        self.context = self.engine.create_execution_context()
        self.buffers = allocate_buffers(self.engine, 1)
        if not self.video.isOpened():
            print("Could not open feed")

    def __del__(self):
        self.video.release()

    """ читаем видео, уменьшаем его и детектим. фун-я flow берет первый кадр, 
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
            self.cropped_frame = old_detect(self.cropped_frame, height, width, 80, self.context, self.buffers)
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



def GiB(val):
    return val*1 << 30


def find_sample_data(description="Runs a TensorRT Python sample", subfolder="", find_files=[]):
    '''
    Parses sample arguments.
    Args:
        description (str): Description of the sample.
        subfolder (str): The subfolder containing data relevant to this sample
        find_files (str): A list of filenames to find. Each filename will be replaced with an absolute path.
    Returns:
        str: Path of data directory.
    Raises:
        FileNotFoundError
    '''

    # Standard command-line arguments for all samples.
    kDEFAULT_DATA_ROOT = os.path.join(os.sep, "usr", "src", "tensorrt", "data")
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--datadir", help="Location of the TensorRT sample data directory.",
                        default=kDEFAULT_DATA_ROOT)
    args, unknown_args = parser.parse_known_args()

    # If data directory is not specified, use the default.
    data_root = args.datadir
    # If the subfolder exists, append it to the path, otherwise use the provided path as-is.
    subfolder_path = os.path.join(data_root, subfolder)
    data_path = subfolder_path
    if not os.path.exists(subfolder_path):
        print("WARNING: " + subfolder_path + " does not exist. Trying " + data_root + " instead.")
        data_path = data_root

    # Make sure data directory exists.
    if not (os.path.exists(data_path)):
        raise FileNotFoundError(data_path + " does not exist. Please provide the correct data path with the -d option.")

    # Find all requested files.
    for index, f in enumerate(find_files):
        find_files[index] = os.path.abspath(os.path.join(data_path, f))
        if not os.path.exists(find_files[index]):
            raise FileNotFoundError(
                find_files[index] + " does not exist. Please provide the correct data path with the -d option.")

    return data_path, find_files


# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine, batch_size):
    inputs = []
    outputs = []
    bindings = []
    # Stream = namedtuple('Stream', ['ptr'])
    # stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)
    stream = cuda.Stream()
    for binding in engine:

        size = trt.volume(engine.get_binding_shape(binding))*batch_size
        dims = engine.get_binding_shape(binding)

        # in case batch dimension is -1 (dynamic)
        if dims[0] < 0:
            size *= -1

        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


TRT_LOGGER = trt.Logger()


def main(engine_path, image_path, image_size):
    with get_engine(engine_path) as engine, engine.create_execution_context() as context:
        buffers = allocate_buffers(engine, 1)
        IN_IMAGE_H, IN_IMAGE_W = image_size
        context.set_binding_shape(0, (1, 3, IN_IMAGE_H, IN_IMAGE_W))

        image_src = cv2.imread(image_path)

        num_classes = 80

        for i in range(2):  # This 'for' loop is for speed check
            # Because the first iteration is usually longer
            boxes = detect(context, buffers, image_src, image_size, num_classes)

        if num_classes == 20:
            namesfile = 'data/voc.names'
        elif num_classes == 80:
            namesfile = 'data/coco.names'
        else:
            namesfile = 'data/names'

        class_names = load_class_names(namesfile)
        plot_boxes_cv2(image_src, boxes[0], savename='predictions_trt.jpg', class_names=class_names)


def get_engine(engine_path):
    # If a serialized engine exists, use it instead of building an engine.
    print("Reading engine from file {}".format(engine_path))
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())



def old_detect(frame, height, width, num_classes, context, buffers):
    overlay = frame.copy()
    ta = time.time()
    """Переменные"""
    busy_car_places_id = np.array([])
    IN_IMAGE_H, IN_IMAGE_W = height, width
    free_space = 0

    """Основной блок, тут детектятся машинки"""
    sized = cv2.resize(frame, (IN_IMAGE_W, IN_IMAGE_H))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
    """Блок из tensorrt"""
    sized = np.transpose(sized, (2, 0, 1)).astype(np.float32)
    sized = np.expand_dims(sized, axis=0)
    sized /= 255.0
    sized = np.ascontiguousarray(sized)
    print("Shape of the network input: ", sized.shape)
    # print(img_in)

    inputs, outputs, bindings, stream = buffers
    print('Length of inputs: ', len(inputs))
    inputs[0].host = sized

    trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

    print('Len of outputs: ', len(trt_outputs))

    trt_outputs[0] = trt_outputs[0].reshape(1, -1, 1, 4)
    trt_outputs[1] = trt_outputs[1].reshape(1, -1, num_classes)

    tb = time.time()

    print('-----------------------------------')
    print('    TRT inference time: %f'%(tb - ta))
    print('-----------------------------------')

    boxes = post_processing(sized, 0.4, 0.6, trt_outputs)


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




def detect(context, buffers, image_src, image_size, num_classes):
    IN_IMAGE_H, IN_IMAGE_W = image_size

    ta = time.time()
    # Input
    resized = cv2.resize(image_src, (IN_IMAGE_W, IN_IMAGE_H), interpolation=cv2.INTER_LINEAR)
    img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
    img_in = np.expand_dims(img_in, axis=0)
    img_in /= 255.0
    img_in = np.ascontiguousarray(img_in)
    print("Shape of the network input: ", img_in.shape)
    # print(img_in)

    inputs, outputs, bindings, stream = buffers
    print('Length of inputs: ', len(inputs))
    inputs[0].host = img_in

    trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

    print('Len of outputs: ', len(trt_outputs))

    trt_outputs[0] = trt_outputs[0].reshape(1, -1, 1, 4)
    trt_outputs[1] = trt_outputs[1].reshape(1, -1, num_classes)

    tb = time.time()

    print('-----------------------------------')
    print('    TRT inference time: %f'%(tb - ta))
    print('-----------------------------------')

    boxes = post_processing(img_in, 0.4, 0.6, trt_outputs)

    return boxes


# if __name__ == '__main__':
#     engine_path = sys.argv[1]
#     image_path = sys.argv[2]
#
#     if len(sys.argv) < 4:
#         image_size = (416, 416)
#     elif len(sys.argv) < 5:
#         image_size = (int(sys.argv[3]), int(sys.argv[3]))
#     else:
#         image_size = (int(sys.argv[3]), int(sys.argv[4]))
#
#     main(engine_path, image_path, image_size)

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