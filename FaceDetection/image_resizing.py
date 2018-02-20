"""
Two ways of resizing all images in directory with opencv
"""
import os

import cv2

"""
First
Loop which goes through all files with ".jpg" extension and resize it
"""
for filename in os.listdir(path_to_directory):
    if filename.endswith(".jpg"):
        image = cv2.imread(filename, 1)
        resized_image = cv2.resize(image, (100, 100)) #size in which are images resized 100x100
        cv2.imwrite(filename, resized_image)


"""
Second
I created a list containing the image file paths and then iterated through that list.
The loop reads each image, resizes, displays, waits for the user input key,
closes the window once the key is pressed and then writes the resized image
under the existing file name together with the "resized" prefix.
"""
import glob
images=glob.glob("*.jpg")

for image in images:
    img=cv2.imread(image,0)
    re=cv2.resize(img,(100,100))
    cv2.imshow("Hey",re)
    cv2.waitKey(500)
    cv2.destroyAllWindows()
cv2.imwrite("resized_"+image,re)