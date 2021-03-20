import pyopenpose as op
import cv2
import matplotlib.pyplot as plt

params = dict()
params["model_folder"] = "../openpose/models/"

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

datum = op.Datum()
imageToProcess = cv2.imread("img.jpg")
# cv2.imshow("qq", imageToProcess)
imageToProcess = cv2.cvtColor(imageToProcess, cv2.COLOR_RGB2BGR)

datum.cvInputData = imageToProcess
opWrapper.emplaceAndPop(op.VectorDatum([datum]))

print("Body keypoints: \n" + str(datum.poseKeypoints))
cv2.imwrite("res.jpg", datum.cvOutputData)

