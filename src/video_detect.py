import cv2 as cv
import os

def detect_faces_in_video():

    # video path
    video_path = os.path.join(os.getcwd(), "video", "vid1.mp4")

    # load the haarcascade for face detection
    cascade_path = os.path.join(cv.data.haarcascades, 'haarcascade_frontalface_default.xml')
    haar_cascade = cv.CascadeClassifier(cascade_path)

    # capture video from file
    cap = cv.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 🔥 Resize the frame to a moderate size (reduce large video)
        frame = cv.resize(frame, (500, 500))  # You can change to (640, 480)

        # convert to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # detect faces
        face_rects = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        print(f'Number of faces detected in frame: {len(face_rects)}')

        # draw rectangles
        for (x, y, w, h) in face_rects:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # show video
        cv.imshow('Video Face Detection', frame)

        # press q to quit
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()











# import cv2 as cv;
# import os;


# def detect_faces_in_video():


#     #image path
#     video_path = os.path.join(os.getcwd(),"video", "vid1.mp4");

#     # load the haarcascade for face detection
#     cascade_path = os.path.join(cv.data.haarcascades, 'haarcascade_frontalface_default.xml');
#     haar_cascade = cv.CascadeClassifier(cascade_path);

#     # capture video from file
#     cap = cv.VideoCapture(video_path);

#     if not cap.isOpened():
#         print("Error: Could not open video.");
#         return

#     while True:
#         ret, frame = cap.read();
#         if not ret:
#             break;

#         gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY);

#         face_rects = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3);

#         print(f'Number of faces detected in frame: {len(face_rects)}');

#         for (x, y, w, h) in face_rects:
#             cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2);

#         cv.imshow('Video Face Detection', frame);

#         if cv.waitKey(1) & 0xFF == ord('q'):
#             break;

#     cap.release();
#     cv.destroyAllWindows();