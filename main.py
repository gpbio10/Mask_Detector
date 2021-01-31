
import cv2
import sys

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
nose_cascade = cv2.CascadeClassifier('CascadeFiles_haarcascade_mcs_nose.xml')
mouth_cascade = cv2.CascadeClassifier('Mouth.xml')

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    noses = nose_cascade.detectMultiScale(gray, 1.3, 5)
    mouths = mouth_cascade.detectMultiScale(gray, 1.7, 5)

    # Draw a rectangle around the faces
#    for (x, y, w, h) in noses:
#        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
#        cv2.putText(frame, "Nose", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0))

#    for (x, y, w, h) in mouths:
#        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
#        cv2.putText(frame, "Mouth", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0))

    nose = False
    mouth = False


    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0))
        for (x_nose, y_nose, w_nose, z_nose) in noses:
            if x <= x_nose and x+w >= x_nose and y <= y_nose and y+h >= y_nose:
                nose = True


        for (x_mouth, y_mouth, w_mouth, z_mouth) in mouths:
            if x <= x_mouth and x+w >= x_mouth and y <= y_mouth and y+h >= y_mouth:
                mouth = True

        if mouth and nose:
            cv2.putText(frame, "NO MASK", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
        else:
            cv2.putText(frame, "MASK", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))




    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()