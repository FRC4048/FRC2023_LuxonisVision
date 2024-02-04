#!/usr/bin/env python3
# Run pip install depthai and pip install robotpy to automatically get the packages for Network Tables
#Also make sure to run pip cache purge to clear the cache files or else pip might download using the cached files
#/older versions.
from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import time
import robotpy
import ntcore

#Network Table Instance
inst = ntcore.NetworkTableInstance.getDefault()
inst.startClient4("Luxonis Client")
inst.setServerTeam(4048)
table = inst.getTable("Luxonis")
inst.startDSClient()
dblTopic = inst.getDoubleTopic("/datatable/Luxonis")
stringTopic = inst.getStringTopic("/datatable/Luxonis")
#returns a list of topics[x,y,z,fps,probability,label]
def publisher():
    xPub = table.getDoubleTopic("x").publish()
    yPub = table.getDoubleTopic("y").publish()
    zPub = table.getDoubleTopic("z").publish()
    fpsPub = table.getDoubleTopic("fps").publish()
    cnfPub = table.getDoubleTopic("prob").publish()
    labelPub = table.getStringTopic("label").publish()
    return [xPub,yPub,zPub,fpsPub,cnfPub,labelPub]
topics = publisher()
datatable = inst.getTable("datatable/Luxonis")

# Get argument first
nnBlobPath = str((Path(__file__).parent / Path('best_openvino_2022.1_6shave.blob')).resolve().absolute())

if not Path(nnBlobPath).exists():
    import sys
    raise FileNotFoundError(f'Required file/s not found, please run "{sys.executable} install_requirements.py"')

# Tiny yolo v3/4 label texts
labelMap = [
    "note"
]

syncNN = True

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
spatialDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
nnNetworkOut = pipeline.create(dai.node.XLinkOut)

xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutNN = pipeline.create(dai.node.XLinkOut)
xoutDepth = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")
xoutNN.setStreamName("detections")
xoutDepth.setStreamName("depth")
nnNetworkOut.setStreamName("nnNetwork")


# Properties
camRgb.setPreviewSize(416,416)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# setting node configs
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
# Align depth map to the perspective of RGB camera, on which inference is done
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
stereo.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())

spatialDetectionNetwork.setBlobPath(nnBlobPath)
spatialDetectionNetwork.setConfidenceThreshold(0.70)
spatialDetectionNetwork.input.setBlocking(False)
spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
spatialDetectionNetwork.setDepthLowerThreshold(100)
spatialDetectionNetwork.setDepthUpperThreshold(5000)

# Yolo specific parameters
spatialDetectionNetwork.setNumClasses(1)
spatialDetectionNetwork.setCoordinateSize(4)
spatialDetectionNetwork.setAnchors([10.0,
                13.0,
                16.0,
                30.0,
                33.0,
                23.0,
                30.0,
                61.0,
                62.0,
                45.0,
                59.0,
                119.0,
                116.0,
                90.0,
                156.0,
                198.0,
                373.0,
                326.0])
spatialDetectionNetwork.setAnchorMasks({ "side52": [0,1,2],"side26": [3,4,5],"side13": [6,7,8]})
spatialDetectionNetwork.setIouThreshold(0.5)

# Linking
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

camRgb.preview.link(spatialDetectionNetwork.input)
if syncNN:
    spatialDetectionNetwork.passthrough.link(xoutRgb.input)
else:
    camRgb.preview.link(xoutRgb.input)

spatialDetectionNetwork.out.link(xoutNN.input)

stereo.depth.link(spatialDetectionNetwork.inputDepth)
spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)
spatialDetectionNetwork.outNetwork.link(nnNetworkOut.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    networkQueue = device.getOutputQueue(name="nnNetwork", maxSize=4, blocking=False);

    startTime = time.monotonic()
    counter = 0
    fps = 0
    color = (255, 255, 255)
    printOutputLayersOnce = True

    while True:
        inPreview = previewQueue.get()
        inDet = detectionNNQueue.get()
        depth = depthQueue.get()
        inNN = networkQueue.get()

        confidenceInterval = 0.65
        confidence = 0.0
        closest_x = 0.0
        closest_y = 0.0
        closest_z = 0.0
        label = "none"

        if printOutputLayersOnce:
            toPrint = 'Output layer names:'
            for ten in inNN.getAllLayerNames():
                toPrint = f'{toPrint} {ten},'
            print(toPrint)
            printOutputLayersOnce = False;
        #Added this code
        # frame = inPreview.getCvFrame()
        # depthFrame = depth.getFrame()  # depthFrame values are in millimeters
        # depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        # depthFrameColor = cv2.equalizeHist(depthFrameColor)
        # depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

        counter+=1
        current_time = time.monotonic()
        if (current_time - startTime) > 1 :
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time
        publisher()[3].set(fps)
        detections = inDet.detections

        # height = frame.shape[0]
        # width = frame.shape[1]
        # If the frame is available, draw bounding boxes on it and show the frame
        if (len(detections) > 0):
            closest_detection = detections[0]
            for detection in detections:
            #all topics are then converted into meters
                 if (closest_detection.spatialCoordinates.z/1000 > detection.spatialCoordinates.z/1000):
                      closest_detection = detection
            confidence = detection.confidence
            closest_x = closest_detection.spatialCoordinates.x/1000
            closest_y = closest_detection.spatialCoordinates.y/1000
            closest_z = closest_detection.spatialCoordinates.z/1000
            # try:
            #    label = str(labelMap[closest_detection.label])
            # except:
            #    label = str(closest_detection.label)


            # roiData = detection.boundingBoxMapping
            # roi = roiData.roi
            # roi = roi.denormalize(depthFrameColor.shape[1], depthFrameColor.shape[0])
            # topLeft = roi.topLeft()
            # bottomRight = roi.bottomRight()
            # xmin = int(topLeft.x)
            # ymin = int(topLeft.y)
            # xmax = int(bottomRight.x)
            # ymax = int(bottomRight.y)
            # cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)
            #
            # # Denormalize bounding box
            # x1 = int(detection.xmin * width)
            # x2 = int(detection.xmax * width)
            # y1 = int(detection.ymin * height)
            # y2 = int(detection.ymax * height)
            # try:
            #     label = labelMap[detection.label]
            # except:
            #     label = detection.label
            # cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            # cv2.putText(frame, "{:.2f}".format(detection.confidence * 100), (x1 + 10, y1 + 35),
            #                 cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            # cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x) / 1000} m", (x1 + 10, y1 + 50),
            #                 cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            # cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y) / 1000} m", (x1 + 10, y1 + 65),
            #                 cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            # cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z) / 1000} m", (x1 + 10, y1 + 80),
            #                 cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            # cv2.putText(frame,f"FPS: {int(fps)}",(x1 + 10, y1 + 95),cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            #
            # cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)
            #
            # cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)
            # cv2.imshow("depth", depthFrameColor)
            # cv2.imshow("rgb", frame)
        #returns a list of topics[x,y,z,fps,probability,label]
        # x, y, z are given in meters and measure the distance from the camera
        publisher()[0].set(closest_x)
        publisher()[1].set(closest_y)
        publisher()[2].set(closest_z)
        publisher()[5].set(label)
        publisher()[4].set(confidence)

        if cv2.waitKey(1) == ord('q'):
            break
