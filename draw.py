import collections
import os
import numpy as np
from PIL import Image

IMAGE_PATHS = ['shit.png']
IMAGES = [os.path.join('data', image) for image in IMAGE_PATHS]

def init():
    image_map = {}
    for image_path in IMAGES:
        image = Image.open(image_path)
        (im_width, im_height) = image.size
        image_np = np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
        image_map[image_path[5:]] = image_np

IMAGE_MAP = init()

def draw_shit_on_image_array(image,
                             boxes,
                             classes,
                             scores,
                             classes_to_replace={44: 'shit.png',
                                                 47: 'shit.png'},
                             max_boxes_to_draw=20,
                             min_score_threshold=.5):
    box_to_draw_map = collections.defaultdict(list)
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_threshold:
            if classes[i] in classes_to_replace:
                box = tuple(boxes[i].tolist())
                box_to_draw_map[box] = classes_to_replace[classes[i]]

    im_width, im_height, _ = image.shape
    for box, replace_image in box_to_draw_map.items():
        xmin, ymin, xmax, ymax = box
        ymin = int(ymin * im_height)
        ymax = int(ymax * (im_height - 1))
        xmin = int(xmin * im_width)
        xmax = int(xmax * (im_width - 1))
        for y in range(ymin, ymax):
            for x in range(xmin, xmax):
                image[x, y, 0] = 0
                image[x, y, 1] = 0
                image[x, y, 2] = 0
