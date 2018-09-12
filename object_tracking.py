import numpy as np
import cv2
import os
import time
import progressbar
import pandas
import sys
import argparse

from otv import video

from YOLO import YOLO_small_tf


def still_image_YOLO(frame_list):
    print "Start DET Phrase"
    yolo = YOLO_small_tf.YOLO_TF()
    progress = progressbar.ProgressBar(
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), ' ', progressbar.ETA()])
    det_frames = []
    for i in progress(range(len(frame_list))):
        yolo.disp_console = False
        result = yolo.detect_from_cvmat(frame_list[i][1])
        det_frames.append(result)
    return det_frames


def main():
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True, type=str)
    parser.add_argument('--perc', default=5, type=int)
    parser.add_argument('--output_name', default='output.mp4', type=str)
    args = parser.parse_args()

    frame_list = video.extract_frames(args.video, args.perc)
    det_frame_list = still_image_YOLO(frame_list)
    video.make_video_from_frames(det_frame_list, args.output_name)

    end = time.time()
    print "running time %d" % (end - start)


if __name__ == '__main__':
    main()