import cv2
import sys
import time

# An openCV project that uses face and other facial features recognition
# to determine if a mask is being worn by a user in the WebCam.
# It works by finding a face and then seeing if both the nose and mouth
# are recognizable. If not, then the computer concludes a mask is being
# worn.


# Import the xml files for recognition of face, mouth and nose
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
nose_cascade = cv2.CascadeClassifier('CascadeFiles_haarcascade_mcs_nose.xml')
mouth_cascade = cv2.CascadeClassifier('Mouth.xml')

# Initialize video capture through WebCam
video_capture = cv2.VideoCapture(0)


# Time reset function to create a delay and smooth out the decision of if a mask is being worn
def time_reset(timing):
    current_time = time.time()
    timing = current_time - timing  # Calculates change in time
    return timing


# Main Function
def main():
    # Reset all times for when the nose and mouth were last seen
    nose_start_time = time.time()
    mouth_start_time = time.time()

    nose_start_time = time_reset(nose_start_time)
    mouth_start_time = time_reset(mouth_start_time)

    # Main loop that will run the recognition
    while True:
        # Capture the current frame from the WebCam
        ret, frame = video_capture.read()

        # Convert to grayscale for recognition purposes
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect for any faces, noses and mouths in the current
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        noses = nose_cascade.detectMultiScale(gray, 1.3, 5)
        mouths = mouth_cascade.detectMultiScale(gray, 1.7, 5)

        # These can be uncommented to see visual recognition of the mouth and nose

        # Draw a rectangle around the noses
        #    for (x, y, w, h) in noses:
        #        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        #        cv2.putText(frame, "Nose", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0))

        # Draw a rectangle around the mouths
        #    for (x, y, w, h) in mouths:
        #        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        #        cv2.putText(frame, "Mouth", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0))

        # Set initial value of mouth and nose to True to represent No Mask
        nose = True
        mouth = True

        # All the coordinates of the faces found in the frame
        for (x, y, w, h) in faces:
            # All the coordinates of the noses found in the frame
            for (x_nose, y_nose, w_nose, z_nose) in noses:
                # Check if the nose that was found is inside of the face that was found (Apart of the same face)
                if x <= x_nose and x + w >= x_nose and y <= y_nose and y + h >= y_nose:
                    # Set nose to False (Hidden by the mask)
                    nose = False
                    # Reset time since nose was found to zero
                    nose_start_time = time_reset(nose_start_time)

            # Loop through all mouths found
            for (x_mouth, y_mouth, w_mouth, z_mouth) in mouths:
                # Checks if mouth was inside of the face found in the frame
                if x <= x_mouth and x + w >= x_mouth and y <= y_mouth and y + h >= y_mouth:
                    # Sets mouth to covered
                    mouth = False
                    mouth_start_time = time_reset(mouth_start_time)

            # Checks when the mouth and nose was last seen and that they are currently hidden
            if mouth_start_time > 2 and nose_start_time > 2 and mouth and nose:
                # Print text that Mask is ON in green
                cv2.putText(frame, "MASK", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
                # Print green frame around face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # If mouth and nose were found
            else:
                # Print No Mask in Red
                cv2.putText(frame, "NO MASK", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
                # Print frame in Red
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        # Press q to stop video
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close window
    video_capture.release()
    cv2.destroyAllWindows()

    return


# Run main

main()
