import cv2
import dlib

predictor = dlib.shape_predictor("/Users/jonathankarun/Documents/projects/shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()

capture = cv2.VideoCapture(1)

while True:
    ret, frame = capture.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        for i in range(68):
            x, y = landmarks.part(i).x, landmarks.part(i).y
            cv2.circle(frame, (x,y), 2, (0, 255, 0), -1)

    cv2.imshow('Facial Landmark Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()