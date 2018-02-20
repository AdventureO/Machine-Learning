import cv2

# Read a video from web camera
video = cv2.VideoCapture(0) # 0,1 or 2 - index of you camera

while True:
    #Read image from video every frame
    #Boolean and numpyarray - first frame of video
    check, frame = video.read()
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Capturing", gray_image)


    key = cv2.waitKey(100)
    if key == ord("q"):
        break


video.release()
cv2.destroyAllWindows()