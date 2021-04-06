import cv2
import time
from time import ctime
from flask import Flask, Response

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
        start_time = time.time()
        self.success, self.frame = self.video.read()
        fps = 1/(time.time() - start_time)
        print(f"Time: {ctime(start_time)} FPS : {fps}")
        while self.success:
            start_time = time.time()
            self.frame = cv2.resize(self.frame, (int(self.frame.shape[1]//2.4), int(self.frame.shape[0]//2.4)))
            fps = 1/(time.time() - start_time)
            print(f"Time: {ctime(start_time)} FPS : {fps}")
            return cv2.imencode('.jpg', self.frame())[1].tobytes()


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