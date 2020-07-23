import face_recognition
import cv2
from imutils.video import FPS
from imutils.video import WebcamVideoStream
import numpy as np
import argparse
import imutils
import time
import os


ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--tolerance", type=float, default=0.6,
	help="Tolerance for face detection")
ap.add_argument("-v", "--video", required=False,
	help="Supply a video")
ap.add_argument("-s", "--source", type=int, required=False,
	help="Choose video source, 0 default")
args = vars(ap.parse_args())

print("[INFO] Parsing images, might take a while.")

objlist = []
obj_img = []
obj_enc = []
obj_names = []
count = 0

for root, dirs, files in os.walk(os.getcwd()):
    for file in files:
        if file.endswith('.jpg'):
            result = ''.join([i for i in str(file.replace(".jpg","")) if not i.isdigit()])
            obj_names.append(result)
            objlist.append(file)
            try:
                obj_img.append(face_recognition.load_image_file(objlist[count]))
                obj_enc.append(face_recognition.face_encodings(obj_img[count])[0])
                print("Image accepted " + file)
            except:
                print("Face not found in {0}".format(file))
                pass
            count = count +1

# Create arrays of known face encodings and their names
known_face_encodings = obj_enc
known_face_names = obj_names


# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

#won't detect anymore, don't need anymore
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "table",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "Screen"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')
print("[INFO] starting video stream...")
#0 for laptop cam, 1... for else
#video_capture = VideoStream(src=1).start() #webcam by default, I use webcam for 360d testing bros

isCV = False

if args["video"] is not None:
    video_capture = cv2.VideoCapture(args["video"])
    isCV = True
else:
    if args["source"] is not None:
        video_capture = WebcamVideoStream(src=args["source"]).start()
    else:
        video_capture = WebcamVideoStream(src=0).start()
    isCV = False

time.sleep(2.0)
fps = FPS().start()

while True:

    if isCV:
        ret, frame = video_capture.read()
    else:
        frame = video_capture.read()
    # Resize frame of video to 1/4 size for faster face recognition processing
    #less cpu/gpu load
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)

	# pass the blob through the network and obtain the detections and
	# predictions
    #conf
    net.setInput(blob)
    detections = net.forward()

	# loop over the detections
    count = 0
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
		# the prediction
        confidence = detections[0, 0, i, 2]
        #its complex mate

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
        if confidence > args["confidence"]:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            label = "{}: {:.2f}%".format(CLASSES[idx],
				confidence * 100)
            labelx = CLASSES[idx]
            cv2.rectangle(frame, (startX, startY), (endX, endY),
				COLORS[idx], 2)
            count = count +1
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
            print("{0} objects in frame, {1} in frame".format(count,labelx))
            



    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Process consecutive frames
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, args["tolerance"])
            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom),  (165, 77, 50), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 5), (right, bottom), (165, 77, 50), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, top - 6), font, 1.0, (255, 255, 255), 1)

    fps.update()
    fps.stop()
    # print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    # print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    cv2.putText(frame, "{:.2f}fps".format(fps.fps()), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release handle to the webcam
try:
    video_capture.stop()
except:
    video_capture.release()

cv2.destroyAllWindows()
