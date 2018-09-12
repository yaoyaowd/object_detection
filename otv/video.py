import cv2
import progressbar


def extract_frames(input_path, input_perc):
    vidcap = cv2.VideoCapture(input_path)
    if not vidcap.isOpened():
        print "Could not open video", input_path
        return
    print "Open video:", input_path
    image = vidcap.read()
    total = int((vidcap.get(cv2.CAP_PROP_FRAME_COUNT)/100) * input_perc)
    print "%d frames to read" % total
    progress = progressbar.ProgressBar(
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), ' ', progressbar.ETA()])

    frames = []
    for i in progress(range(0, total)):
        frames.append(image)
        image = vidcap.read()
    print "Finish reading video:", input_path
    return frames


def make_video_from_frames(frames, output_path):
    img = cv2.imread(frames[0], True)
    h, w = img.shape[:2]
    fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (w, h), True)
    print("Start Making File Video:%s " % output_path)
    print("%d Frames to Compress" % len(frames))
    progress = progressbar.ProgressBar(
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), ' ', progressbar.ETA()])
    for i in progress(range(len(output_path))):
        out.write(img)
        img = cv2.imread(frames[i], True)
    out.release()
    print("Finished Making File Video:%s " % output_path)