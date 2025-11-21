import cv2 as cv;
import os;


def detect_face():

    #image path
    img_path = os.path.join(os.getcwd(),"images", "group1.jpg");
    img = cv.imread(img_path);
    cv.imshow('original image', img);

    # convert to gray scale because haarcascade works on gray scale images
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY);
    cv.imshow('gray image', gray);

    # load the haarcascade for face detection
    cascade_path = os.path.join(cv.data.haarcascades, 'haarcascade_frontalface_default.xml');

    haar_cascade = cv.CascadeClassifier(cascade_path);

    face_rects =  haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1);

    print(f'Number of faces detected: {len(face_rects)}');

    # draw rectangle around the faces detected 
    for (x, y, w, h) in face_rects:
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2);
    
    cv.imshow('detected faces', img);


    cv.waitKey(0);
    cv.destroyAllWindows();



# next project
# Detect face in video and tell us how many faces are there in each frame
# Detect face in a webcam feed and tell us how many faces are there in the frame

