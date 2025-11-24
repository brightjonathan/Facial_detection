import os;
import cv2 as cv;
import numpy as np;

DIR = r'C:\Users\User\Desktop\FaceDetection\face_detection\Faces\train';
people =  [];
for i in os.listdir(DIR): 
    people.append(i);
# print(people);

def create_train_data():

    # FEATURES: images array of the faces;
    # LABELS: who faces does it belong to;
    features=[]; 
    labels=[];

    for person in people:
        path = os.path.join(DIR, person);
        label = people.index(person);
        for img in os.listdir(path):
            img_path = os.path.join(path, img);
            img_array = cv.imread(img_path);
            if img_array is None:
                continue;
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY);

             # load the haarcascade for face detection
            cascade_path = os.path.join(cv.data.haarcascades, 'haarcascade_frontalface_default.xml');

            haar_cascade = cv.CascadeClassifier(cascade_path);
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4);
            for (x, y, w, h) in faces_rect:
                face_roi = gray[y:y+h, x:x+w];
                features.append(face_roi);
                labels.append(label); 

    # print(len(feartures));
    # print(len(labels));

    print('Training done ---------------');

    # Convert features and labels to numpy arrays why?: because opencv works with numpy arrays
    features = np.array(features, dtype='object');
    labels = np.array(labels);

    # Local Binary Patterns Histograms Face Recognizer (LBPH);
    face_recognizer = cv.face.LBPHFaceRecognizer_create();

    # Train the Recognizer on the features list and the labels list
    face_recognizer.train(features,labels);

    face_recognizer.save('face_trained.yml');
    np.save('features.npy', features);
    np.save('labels.npy', labels);
    



