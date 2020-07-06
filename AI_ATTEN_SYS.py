import xlrd
import cv2
import pyzbar
from pyzbar.pyzbar import decode
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import os
import smtplib as s
def mailsend(d,ind):
    server=s.SMTP('smtp.gmail.com',587)
    server.ehlo()
    server.starttls()
    server.login('rkordruas@gmail.com','SauR123?')
    server.sendmail('rkordruas@gmail.com',cle[ind],'your attendance is '+d+'%')
    server.close()
    print('mail sent successfully to ',cle[ind])

fd=cv2.CascadeClassifier(r'C:\Users\sayan\Desktop\AI Project\face-mask-caffe\haarcascade_frontalface_alt2.xml')
rb=xlrd.open_workbook(r'C:\Users\sayan\Desktop\AI Project\db.xls')
print(rb.nsheets)
sh=rb.sheet_by_name('info')
rn=sh.nrows
print(rn)
for i in range(0,rn):
    r=sh.row_values(i)
    print(r)
cl=sh.col_values(4)
cle=sh.col_values(2)
print(cl)
from xlutils.copy import copy
wbb=copy(rb)
shd0=wbb.get_sheet(0)
t=time.ctime()
td=t.split(' ')
pos=int(td[3])
tdt=td[4].split(':')
ct=int(tdt[0])
def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			face = np.expand_dims(face, axis=0)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		preds = maskNet.predict(faces)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNetFromCaffe(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
flag=1
imgcount=0
# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    if(pos==31) and (ct>=9) and (flag==1):
        for mid in range(1,rn):
            count=0
            for loc in range(5,36):
                rb=xlrd.open_workbook(r'C:\Users\sayan\Desktop\AI Project\db.xls')
                sh=rb.sheet_by_name('info')
                crv=sh.row_values(mid)
                if(crv[loc]=='P'):
                    count=count+1
            d=str((count/31)*100)
            print(d,mid)
            mailsend(d,mid)
        flag=0
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    
    # detect faces in the frame and determine if they are wearing a
    # face mask or not
    if(imgcount>1):
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        # loop over the detected face locations and their corresponding
        # locations
        for (box, pred) in zip(locs, preds):
                # unpack the bounding box and predictions
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred

                # determine the class label and color we'll use to draw
                # the bounding box and text
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                # include the probability in the label
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
                print(label)
                # display the label and bounding box rectangle on the output
                # frame
                cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 5)
                if(label.split(':')[0]=='Mask'):
                    dqr=decode(frame)
                    if(len(dqr)>0):
                        d=dqr[0].data.decode()
                        print(d)
                        res=d in cl
                        ind=cl.index(d)
                        cellv=sh.cell_value(ind,3+pos)
                        if(cellv=='P'):
                            print('Your Attendance is already marked.', 'Thank you')
                        else:
                            shd0.write(ind,4+pos,'P')
                            wbb.save(r'C:\Users\sayan\Desktop\AI Project\db.xls')
                            print('Your attendance is marked', 'Thank You')
                    else:
                        print('Please show your QR ID')
                else:
                    print('Please wear Mask to mark your attendance')
        imgcount=0
    imgcount=imgcount+1 
    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the esc key was pressed, break from the loop
    if key == 27:
            break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
