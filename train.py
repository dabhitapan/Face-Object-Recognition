import cv2
import argparse
import os, errno


ap = argparse.ArgumentParser()
ap.add_argument("-n", "--name",required=True,
	help="Name of person to train")
args = vars(ap.parse_args())

strx = str(args["name"])
try:
    os.makedirs(strx)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

vidcap = cv2.VideoCapture(0);
success,image = vidcap.read()
count = 0
success = True

while success:
    success,image = vidcap.read()
    print('read a new frame:',success)
    if count%10 == 0 :
         cv2.imwrite(strx +'/'+ strx +'%d.jpg'%count,image)
         print('success')
    count+=1