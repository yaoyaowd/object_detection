import collections
import os
import numpy as np
from PIL import Image

IMAGE_PATHS = ['shit.jpg']
IMAGES = [os.path.join('data', image) for image in IMAGE_PATHS]

def init():
    image_map = {}
    for image_path in IMAGES:
        image = Image.open(image_path)
        (im_width, im_height) = image.size
        image_np = np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
        image_map[image_path[5:]] = image_np
    return image_map

IMAGE_MAP = init()

def draw_shit_on_image_array(image,
                             boxes,
                             classes,
                             scores,
                             classes_to_replace={44: 'shit.jpg',
                                                 47: 'shit.jpg'},
                             max_boxes_to_draw=20,
                             min_score_threshold=.5):
    box_to_draw_map = collections.defaultdict(list)
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_threshold:
            if classes[i] != 1:
                box = tuple(boxes[i].tolist())
                box_to_draw_map[box] = 'shit.jpg'

    im_width, im_height, _ = image.shape
    for box, replace_image in box_to_draw_map.items():
        xmin, ymin, xmax, ymax = box
        ymin = int(ymin * im_height)
        ymax = int(ymax * (im_height - 1))
        xmin = int(xmin * im_width + 0.05 * im_width)
        replace_image = IMAGE_MAP[replace_image]
        ri_width, ri_height, _ = replace_image.shape
        size = ymax - ymin
        for x in range(xmin - size, xmin):
            for y in range(ymin, ymax):
                if x >= 0 and x < im_width:
                    xx = int(float(x - xmin + size) / size * ri_height)
                    yy = int(float(y - ymin) / size * ri_width)
                    if replace_image[xx, yy, 0] != 255 \
                            or replace_image[xx, yy, 1] != 255 \
                            or replace_image[xx, yy, 2] != 255:
                        image[x,y] = replace_image[xx,yy]
