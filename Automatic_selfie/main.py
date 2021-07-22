import cv2
import datetime

# define a video capture object
cap = cv2.VideoCapture(0)

# adding the cascade files
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')


# Capture the video frame by frame
while True:
    ret,frame = cap.read()

    # creating a copy of the frame so that the rectangle would not be there in the captured image
    original_frame=frame.copy()

    # converting original to gray image
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # detecting the face using the cascade file
    face = face_cascade.detectMultiScale(gray,1.3,5)
    # drawing the rectangle around the face after detection
    for x,y,w,h in face:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

        # getiing only the region of interest from the gray and original
        face_roi = frame[y:y+h,x:x+w]
        gray_roi = gray[y:y+h,x:x+w]
        smile=smile_cascade.detectMultiScale(gray_roi,1.3,25)

        # drawing rectangle around the smile of the face
        for x1, y1, w1, h1 in smile:
            cv2.rectangle(face_roi, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 255), 2)
            time_stamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

            file_name = f'selfie-{time_stamp}.png'
            # capturing when smile is detected
            cv2.imwrite(file_name,original_frame)

    # showing the output
    cv2.imshow('gray',gray)
    cv2.imshow('camera',frame)
    # the 'q' button is set as the quitting button
    if cv2.waitKey(10) == ord('q'):
        break