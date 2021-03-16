import pyopenpose as op
import cv2

params = dict()
params["model_folder"] = "./openpose/models/"
params["face"] = True
params["hand"] = True

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

datum = op.Datum()
imageToProcess = cv2.imread("img.jpg")
datum.cvInputData = imageToProcess
opWrapper.emplaceAndPop(op.VectorDatum([datum]))

print("Body keypoints: \n" + str(datum.poseKeypoints))
cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)