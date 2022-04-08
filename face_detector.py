import cv2

# imported pre-trained data for detecting faces
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# image to detect faces
# img = cv2.imread('tom-holland.png')

# to capture video from webcam
webcam = cv2.VideoCapture(0)

# iterate forever over frames
while True:

    # read current frame
    successful_frame_read, frame = webcam.read()

    # convert image to gray
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect Faces
    face_coordinates = trained_face_data.detectMultiScale(gray_image)

    # draw rectangles round face
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 10)

    cv2.imshow('Face Detector', frame)
    key = cv2.waitKey(1)

    # stop if Q key is pressed
    if key == 81 or key == 113:
        break

# release the video
webcam.release()

print("code completed")
