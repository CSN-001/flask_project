import pyopenpose as op
import cv2
import numpy as np

params = dict()
params["model_folder"] = "../openpose/models/"

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

datum = op.Datum()
imageToProcess = cv2.imread("img.jpg")

datum.cvInputData = imageToProcess
opWrapper.emplaceAndPop(op.VectorDatum([datum]))

print("Body keypoints: \n" + str(datum.poseKeypoints))
np.savetxt("res.txt", ',')
# cv2.imwrite("res.jpg", datum.cvOutputData)

