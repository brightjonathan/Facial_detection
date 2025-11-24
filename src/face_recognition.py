import os
import cv2 as cv
import numpy as np

def recognize_face():

    # Load people names
    DIR = r'C:\Users\User\Desktop\FaceDetection\face_detection\Faces\train'
    people = [p for p in os.listdir(DIR)]

    # Load Haarcascade
    cascade_path = os.path.join(cv.data.haarcascades, 'haarcascade_frontalface_default.xml')
    haar_cascade = cv.CascadeClassifier(cascade_path)

    # Load trained model + data
    features = np.load('features.npy', allow_pickle=True)
    labels = np.load('labels.npy')

    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    face_recognizer.read('face_trained.yml')

    # Load test image
    img_path = r"C:\Users\User\Desktop\FaceDetection\face_detection\Faces\train\Ben Afflek/1.jpg"
    img = cv.imread(img_path)

    if img is None:
        print("ERROR: Failed to load image.")
        return

    # Resize big images (Haarcascade performs better on smaller)
    img = cv.resize(img, (300, 300))

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow("Gray Image", gray)

    # Detect faces (improved parameters)
    faces_rect = haar_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(80, 80)
    )

    print("Faces detected:", len(faces_rect))

    if len(faces_rect) == 0:
        print(" No face detected!")
        cv.imshow("Detected Face", img)
        cv.waitKey(0)
        cv.destroyAllWindows()
        return

    # Process each face found
    for (x, y, w, h) in faces_rect:

        faces_roi = gray[y:y+h, x:x+w]

        # Predict using LBPH
        label, confidence = face_recognizer.predict(faces_roi)

        print(f"Label = {people[label]} with confidence = {confidence}")

        # Optional: If confidence > 120 → ignore (LBPH threshold)
        if confidence > 120:
            name = "Unknown"
        else:
            name = people[label]

        # Draw text
        cv.putText(img, name, (x, y - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw rectangle
        cv.rectangle(img, (x, y), (x + w, y + h),
                     (0, 255, 0), 2)

    cv.imshow("Detected Face", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

