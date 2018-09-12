import cv2
import numpy as np
import time
import tensorflow as tf
import os

from multiprocessing import Queue, Pool
from video.video import FPS, VideoStream
from utils import label_map_util
from utils import visualization_utils

WIDTH = 360
HEIGHT = 240
VIDEO_SOURCE = 0
NUM_WORKER = 2
QUEUE_SIZE = 5
PATH_TO_CKPT = 'ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90
LABEL_MAP = label_map_util.load_labelmap(PATH_TO_LABELS)
CATEGORIES = label_map_util.convert_label_map_to_categories(LABEL_MAP, max_num_classes=NUM_CLASSES, use_display_name=True)
CATEGORY_INDEX = label_map_util.create_category_index(CATEGORIES)


def detect(graph, sess, image_np):
    for i in range(image_np.shape[0]):
        for j in range(image_np.shape[1]):
            image_np[i,j,0], image_np[i,j,2] = image_np[i,j,2], image_np[i,j,0]

    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = graph.get_tensor_by_name("image_tensor:0")
    boxes = graph.get_tensor_by_name('detection_boxes:0')
    scores = graph.get_tensor_by_name('detection_scores:0')
    classes = graph.get_tensor_by_name('detection_classes:0')
    num_detections = graph.get_tensor_by_name('num_detections:0')
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    visualization_utils.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        CATEGORY_INDEX,
        use_normalized_coordinates=True,
        line_thickness=8)

    for i in range(image_np.shape[0]):
        for j in range(image_np.shape[1]):
            image_np[i,j,0], image_np[i,j,2] = image_np[i,j,2], image_np[i,j,0]
    return image_np


def worker(input_q, output_q):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name="")

        with tf.Session(graph=detection_graph) as sess:
            while True:
                try:
                    frame = input_q.get()
                    output_q.put(detect(detection_graph, sess, frame))
                except KeyboardInterrupt:
                    break


if __name__ == '__main__':
    input_q = Queue(maxsize=QUEUE_SIZE)
    output_q = Queue(maxsize=QUEUE_SIZE)
    pool = Pool(processes=NUM_WORKER, initializer=worker, initargs=(input_q, output_q))
    video = VideoStream(src=VIDEO_SOURCE, width=WIDTH, height=HEIGHT).start()
    fps = FPS().start()
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    video_out = None

    while True:
        try:
            frame = video.read()
            if not video_out:
                video_out = cv2.VideoWriter("data/output.avi", fourcc, 1, (frame.shape[1], frame.shape[0]))
            input_q.put(frame)
            t = time.time()
            output_frame = output_q.get()
            video_out.write(output_frame)
            cv2.imshow("Video", output_frame)
            cv2.waitKey(1)
            fps.update()
            print('approx. FPS: {:.2f}'.format(fps.fps()))
        except KeyboardInterrupt:
            break

    print('elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('approx. FPS: {:.2f}'.format(fps.fps()))
    pool.terminate()
    fps.stop()
    video.stop()
    video_out.release()
    cv2.destroyAllWindows()
