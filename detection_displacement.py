import cv2
import numpy as np

from utilities.utilities import out_detection, calculateDistance

capture = cv2.VideoCapture(0)
# Load Yolo
net = cv2.dnn.readNet(r"weights/yolov3_training_last.weights", r"weights/yolov3-tiny.cfg")

getImage1 = cv2.imread(r'images/flowerpot.jpeg')

smallImageoldx1 = 450
smallImageoldy1 = 200

xmiddleImage1 = 20
ymiddleImage1 = 20

smallImage1 = cv2.resize(getImage1, (75, 75))

xlastImage1 = 75 - xmiddleImage1
ylastImage1 = 75 - ymiddleImage1

smallImageoldwidth1 = smallImageoldx1 + xmiddleImage1 + xlastImage1
smallImageoldhigh1 = smallImageoldy1 + ymiddleImage1 + ylastImage1

getImage2 = cv2.imread(r'images/centre-table.jpeg')

smallImageoldx2 = 200
smallImageoldy2 = 200

xmiddleImage2 = 20
ymiddleImage2 = 20

smallImage2 = cv2.resize(getImage2, (130, 130))

xlastImage2 = 130 - xmiddleImage2
ylastImage2 = 130 - ymiddleImage2

smallImageoldwidth2 = smallImageoldx2 + xmiddleImage2 + xlastImage2
smallImageoldhigh2 = smallImageoldy2 + ymiddleImage2 + ylastImage2

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

font = cv2.FONT_HERSHEY_PLAIN

isTouch = 0
while True:
    _, frame = capture.read()

    # touch = 0
    img = cv2.resize(frame, None, fx=1, fy=1)
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    function_response = out_detection(outs, height, width)
    getDistanceFromTopFinger = calculateDistance(function_response[0])
    if getDistanceFromTopFinger[0] < 30:
        touch = 1

    indexes = cv2.dnn.NMSBoxes(function_response[1], function_response[2], 0.6, 0.7)

    for i in range(len(function_response[1])):
        if i in indexes:
            x, y, w, h = function_response[1][i]
            color = (0, 0, 255)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)

    if touch == 1:
        if getDistanceFromTopFinger[1] in range(smallImageoldx1, smallImageoldwidth1) \
                and getDistanceFromTopFinger[2] in range(smallImageoldy1, smallImageoldhigh1):
            if isTouch == 0:
                xmiddleImage1 = getDistanceFromTopFinger[1] - smallImageoldx1
                ymiddleImage1 = getDistanceFromTopFinger[2] - smallImageoldy1
                isTouch = 1

            smallImageoldx1 = getDistanceFromTopFinger[1] - xmiddleImage1
            smallImageoldy1 = getDistanceFromTopFinger[2] - ymiddleImage1

            xlastImage1 = 75 - xmiddleImage1
            ylastImage1 = 75 - ymiddleImage1

            smallImageoldwidth1 = smallImageoldx1 + xmiddleImage1 + xlastImage1
            smallImageoldhigh1 = smallImageoldy1 + ymiddleImage1 + ylastImage1
        if getDistanceFromTopFinger[1] in range(smallImageoldx2, smallImageoldwidth2) \
                and getDistanceFromTopFinger[2] in range(smallImageoldy2, smallImageoldhigh2):
            if isTouch == 0:
                xmiddleImage2 = getDistanceFromTopFinger[1] - smallImageoldx2
                ymiddleImage2 = getDistanceFromTopFinger[2] - smallImageoldy2
                isTouch = 1

            smallImageoldx2 = getDistanceFromTopFinger[1] - xmiddleImage2
            smallImageoldy2 = getDistanceFromTopFinger[2] - ymiddleImage2

            xlastImage2 = 130 - xmiddleImage2
            ylastImage2 = 130 - ymiddleImage2

            smallImageoldwidth2 = smallImageoldx2 + xmiddleImage2 + xlastImage2
            smallImageoldhigh2 = smallImageoldy2 + ymiddleImage2 + ylastImage2

    else:
        isTouch = 0

    try:
        img[smallImageoldy1:smallImageoldhigh1, smallImageoldx1:smallImageoldwidth1] = smallImage1
    except :
        pass

    try:
        img[smallImageoldy2:smallImageoldhigh2, smallImageoldx2:smallImageoldwidth2] = smallImage2
    except :
        pass

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

