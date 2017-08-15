# http://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/
# http://www.pyimagesearch.com/2016/12/19/install-opencv-3-on-macos-with-homebrew-the-easy-way/
# https://medium.com/towards-data-science/building-a-real-time-object-recognition-app-with-tensorflow-and-opencv-b7a2b4ebdc32

import cv2
import datetime
from threading import Thread

class FPS:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.num_frames = 0

    def start(self):
        self.start_time = datetime.datetime.now()
        return self

    def stop(self):
        self.end_time = datetime.datetime.now()

    def update(self):
        self.num_frames += 1

    def elapsed(self):
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        else:
            return (datetime.datetime.now() - self.start_time).total_seconds()

    def fps(self):
        return self.num_frames / float(self.elapsed())


class VideoStream:
    def __init__(self, src, width, height):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            self.grabbed, self.frame = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
