
import cv2
import sys

#cascPath = sys.argv[1]
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()



# import cv2
# import sys
#
#
# cascPath = sys.argv[1]
# faceCascade = cv2.CascadeClassifier(cascPath)
#
# cam = cv2.VideoCapture(0)
# cam.set(3, 1080)
# cam.set(4, 720)
#
# if not cam.isOpened():
#     raise IOError("Cannot open webcam")
#
# while True:
#     ret, frame = cam.read()
#
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     faces = faceCascade.detectMultiScale(
#         gray,
#         scaleFactor=1.1,
#         minNeighbors=5,
#         minSize=(30, 30),
#         flags=cv2.cv.CV_HAAR_SCALE_IMAGE
#     )
#
#     for (x,y,w,h) in faces:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#
#     cv2.imshow("Input", frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cam.release()
# cv2.destroyAllWindows()