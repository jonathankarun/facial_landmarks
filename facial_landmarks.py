import cv2
import dlib

predictor = dlib.shape_predictor("File_Path")
detector = dlib.get_frontal_face_detector()

capture = cv2.VideoCapture(1) 

while True:
    ret, frame = capture.read()
    #If nothing is being captured break out of the while loop
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Uses grayscale to identify landmarks better
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        #Iterate through each landmark and create a circle
        for i in range(68):
            x, y = landmarks.part(i).x, landmarks.part(i).y
            cv2.circle(frame, (x,y), 2, (0, 255, 0), -1)

    #Creates Facial Landmark Detection window
    cv2.imshow('Facial Landmark Detection', frame)

    #User input "q" will break out of the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#Stops recording and gets rid of the window
capture.release()
cv2.destroyAllWindows()
