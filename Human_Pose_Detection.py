import cv2 as cv
import matplotlib.pyplot as plt
import argparse


#Git link to download grapth_opt_pb
# https://github.com/quanhua92/human-pose-estimation-opencv


# Youtube video
# https://www.youtube.com/watch?v=9jQGsUidKHs&t=1333s


# Assign model tensorflo config file
net = cv.dnn.readNetFromTensorflow("/home/vert/human-pose-estimation-opencv-master/graph_opt.pb")

inWidth = 368
inHeight = 368
thr = 0.2

# Just reading the image
img = cv.imread("/home/vert/iCamPlus/images (1).jpg")
#
# cv.imshow("image", img)
# cv.waitKey(0)


# def pose_estimation(frame):
#     frameWidth = frame.shape[1]
#     frameHeight = frame.shape[0]
#     net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
#     out = net.forward()
#     out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements
#
#     assert (len(BODY_PARTS) == out.shape[1])
#
#     points = []
#     for i in range(len(BODY_PARTS)):
#         # Slice heatmap of corresponging body's part.
#         heatMap = out[0, i, :, :]
#
#         # Originally, we try to find all the local maximums. To simplify a sample
#         # we just find a global one. However only a single pose at the same time
#         # could be detected this way.
#         _, conf, _, point = cv.minMaxLoc(heatMap)
#         x = (frameWidth * point[0]) / out.shape[3]
#         y = (frameHeight * point[1]) / out.shape[2]
#         # Add a point if it's confidence is higher than threshold.
#         points.append((int(x), int(y)) if conf > args.thr else None)
#
#     for pair in POSE_PAIRS:
#         partFrom = pair[0]
#         partTo = pair[1]
#         assert (partFrom in BODY_PARTS)
#         assert (partTo in BODY_PARTS)
#
#         idFrom = BODY_PARTS[partFrom]
#         idTo = BODY_PARTS[partTo]
#
#         if points[idFrom] and points[idTo]:
#             cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
#             cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
#             cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
#
#     t, _ = net.getPerfProfile()
#     freq = cv.getTickFrequency() / 1000
#     cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
#
#     return frame


# Provide all requirement for model
parser = argparse.ArgumentParser()
parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')
parser.add_argument('--thr', default=0.2, type=float, help='Threshold value for pose parts heat map')
parser.add_argument('--width', default=368, type=int, help='Resize input to specific width.')
parser.add_argument('--height', default=368, type=int, help='Resize input to specific height.')

# Assign args.
args = parser.parse_args()

# Define body part
BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
              "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
              "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
              "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

# Define body paris
POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
              ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
              ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
              ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
              ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]

# Config height and weidth
inWidth = args.width
inHeight = args.height

# net = cv.dnn.readNetFromTensorflow("graph_opt.pb")

net = cv.dnn.readNetFromTensorflow("/home/vert/human-pose-estimation-opencv-master/graph_opt.pb")


# Reading video from local drive
cap = cv.VideoCapture("/home/vert/iCamPlus/iCamPlus_2021-07-08_171828_1_19_.mp4")
# cap = cv.VideoCapture("/home/vert/iCamPlus/Video.mp4")
# cap.set(3,1200)
# cap.set(4,1200)

# cap.("")
# cap = cv.VideoCapture(args.input if args.input else 0)
# cap = cv.VideoCapture(args.input if args.input else 0)

# Checking video is available in folder or not if not it will give some message.
if not cap.isOpened():
    cap = cv.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open video")

# Passing image from video to detect the pose
while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()

        if not hasFrame:
            cv.waitKey()
            break

        # RESIZING image FRAME to display...
        scale_percent = 150
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)

        # Resize original image to display according to our requirement...
        frame = cv.resize(frame, (width, height), interpolation=cv.INTER_AREA)
        height, width, _ = frame.shape
        size = (width, height)

        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]

        net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
        out = net.forward()
        out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

        assert (len(BODY_PARTS) == out.shape[1])

        points = []
        for i in range(len(BODY_PARTS)):
            # Slice heatmap of corresponging body's part.
            heatMap = out[0, i, :, :]

            # Originally, we try to find all the local maximums. To simplify a sample
            # we just find a global one. However only a single pose at the same time
            # could be detected this way.
            _, conf, _, point = cv.minMaxLoc(heatMap)
            x = (frameWidth * point[0]) / out.shape[3]
            y = (frameHeight * point[1]) / out.shape[2]
            # Add a point if it's confidence is higher than threshold.
            points.append((int(x), int(y)) if conf > args.thr else None)

        for pair in POSE_PAIRS:
            partFrom = pair[0]
            partTo = pair[1]
            assert (partFrom in BODY_PARTS)
            assert (partTo in BODY_PARTS)

            idFrom = BODY_PARTS[partFrom]
            idTo = BODY_PARTS[partTo]

            if points[idFrom] and points[idTo]:
                cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
                cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
                cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

        t, _ = net.getPerfProfile()
        freq = cv.getTickFrequency() / 1000
        cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        cv.imshow('OpenPose using OpenCV', frame)
        # cv.waitKey(0)

