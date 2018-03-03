from datetime import datetime

import cv2
import pandas

# Read a video from web camera
video = cv2.VideoCapture(0) # 0,1 or 2 - index of you camera

first_frame = None
status_list = [None, None]
times = []
df = pandas.DataFrame(columns=["Start", "End"])

while True:
    #Read image from video every frame
    #Boolean and numpy array - first frame of video
    check, frame = video.read()
    status = 0
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Blur the image for calculating difference between frames
    gray_image = cv2.GaussianBlur(gray_image, (21, 21), 0) # default numbers of blurriness, 0 - standart deviation

    #Save first loop
    if first_frame is None:
        first_frame = gray_image
        continue # to the beginning of the loop

    # difference between images
    delta_frame = cv2.absdiff(first_frame, gray_image)

    #values that are more than 30 assign to 255(white)
    thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]

    #smooth treshold area
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    #countours of all white object
    (_,cnts,_) = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue

        status = 1

        #get data of contours
        #print(cv2.boundingRect(contour))
        (x, y, w, h) = cv2.boundingRect(contour)

        #draw contours on frames
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 3)

    #check difference between status
    status_list.append(status)

    if status_list[-1] == 1 and status_list[-2] == 0:
        times.append(datetime.now())
    if status_list[-1] == 0 and status_list[-2] == 1:
        times.append(datetime.now())

    #show images
    #cv2.imshow("First frame", first_frame)
    #cv2.imshow("Gray Frame", gray_image)
    #cv2.imshow("Delta Frame", delta_frame)
    #cv2.imshow("Threshold Frame", thresh_frame)
    cv2.imshow("Color Frame", frame)
    #print(status)

    key = cv2.waitKey(1)

    if key == ord("q"):
        if status == 1:
            times.append(datetime.now())
        break

    #print(status)

#print(status_list)
print(times)


for i in range(0, len(times), 2):
    df = df.append({"Start": times[i], "End":times[i+1]}, ignore_index=True)

print(df)
df.to_csv("Times.csv")

# When everything done, release the capture
video.release()
cv2.destroyAllWindows()