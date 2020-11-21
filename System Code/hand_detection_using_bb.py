import argparse
import cv2

from yolo import YOLO

ap = argparse.ArgumentParser()
ap.add_argument('-n', '--network', default="normal", help='Network Type: normal')
ap.add_argument('-d', '--device', default=0, help='Device to use')
ap.add_argument('-s', '--size', default=416, help='Size for yolo')
ap.add_argument('-c', '--confidence', default=0.2, help='Confidence for yolo')
args = ap.parse_args()

if args.network == "normal":
    print("loading yolo...")
    yolo = YOLO("models/cross-hands.cfg", "models/cross-hands.weights", ["hand"])

while True :
    timecounter = 0
    starter = 0
    ix = 0
    iy = 0
    ixx = 0
    iyy = 0
    t = 0
    arrows = cv2.imread('images/sofa.jpeg')
    sofa = cv2.imread('images/sofa.jpeg')

    yolo.size = int(args.size)
    yolo.confidence = float(args.confidence)

    print("starting webcam...")
    cv2.namedWindow("preview")
    # cv2.setWindowProperty ("preview", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    try:
        vc = cv2.VideoCapture(0)
        if vc.isOpened():  # try to get the first frame
            rval, frame = vc.read()

        else:
            rval = False

        while rval:
            while frame.shape[0] < arrows.shape[0] or frame.shape[1] < arrows.shape[1]:
                scale_percent = 30  # percent of original size
                width = int(arrows.shape[1] * scale_percent / 100)
                height = int(arrows.shape[0] * scale_percent / 100)
                dim = (width, height)
                arrows = cv2.resize(arrows, dim, interpolation=cv2.INTER_AREA)
                sofa = cv2.resize(sofa, dim, interpolation=cv2.INTER_AREA)

            width, height, inference_time, results = yolo.inference(frame)
            for detection in results:
                id, name, confidence, x, y, w, h = detection
                cx = x + (w / 2)
                cy = y + (h / 2)

                # draw a bounding box rectangle and label on the image
                color = (0, 255, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                x_offset = x
                y_offset = y
                if x_offset < 0:
                    x_offset = 0
                elif x_offset + arrows.shape[1] > frame.shape[1]:
                    x_offset = frame.shape[1] - arrows.shape[1]
                if y_offset < 0:
                    y_offset = 0
                elif y_offset + arrows.shape[1] > frame.shape[1]:
                    y_offset = frame.shape[0] - arrows.shape[0]
                if x_offset > frame.shape[1]:
                    x_offset = frame.shape[1] - arrows.shape[1]
                elif y_offset > frame.shape[0]:
                    y_offset = frame.shape[0] - arrows.shape[0]
                frame[y_offset:y_offset + arrows.shape[0], x_offset:x_offset + arrows.shape[1]] = arrows
                text = "%s (%s)" % (name, round(confidence, 2))
                if (starter == 0):
                    starter = x
                elif (starter < x - 30 or starter > x + 30):
                    starter = 0
                    timecounter = 0
                else:
                    timecounter = timecounter + 1

                if (timecounter >= 20):
                    print('Image added, loading next image')
                    ix = x
                    ixx = x
                    iy = y
                    iyy = y
                    timecounter = 0

            if (ix != 0 and iy != 0):
                t = t + 1
                arrows = cv2.imread('images/table.jpeg')
                ix = 0
                iy = 0

            if (t == 1):
                frame[iyy:iyy + sofa.shape[0], ixx:ixx + sofa.shape[1]] = sofa
                cv2.imshow("preview", frame)
            else:
                cv2.imshow('preview', frame)

            rval, frame = vc.read()

            key = cv2.waitKey(20)
            if key == 27:  # exit on ESC
                exit()

        cv2.destroyWindow("preview")
        vc.release()
    except ValueError:
        continue

