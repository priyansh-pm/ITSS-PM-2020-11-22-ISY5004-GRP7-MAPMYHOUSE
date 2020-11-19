import numpy as np
import cv2


def out_detection(outs, height, width):
    confidences = []
    boxes = []
    newTopdata = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.1:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                newTopdata.append([center_x, center_y])
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
    return [newTopdata, boxes, confidences]


def calculateDistance(topLists):
    result = [0, 0, 0]
    if len(topLists) > 1:
        top1x = topLists[0][0]
        top2x = topLists[1][0]
        distance = top1x - top2x
        if top2x > top1x:
            distance = top2x - top1x

        distanceX = int(top1x / 2) + int(top2x / 2)
        distanceY = int(topLists[0][1] / 2) + int(topLists[1][1] / 2)
        result = [distance, distanceX, distanceY]
    return result
