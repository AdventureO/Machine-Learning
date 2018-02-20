import cv2

# PARAMETERS FOR FACE DETECTION
#
#  cascadeClassifier.detectMultiScale(image, faceDetections, scaleFactor, minNeighbors, flags, minSize, maxSize)
#
# 1.) scaleFactor: Since some faces may be closer to the camera, they would appear bigger
# 		 than other faces in the background -> the scale factor compensates for this
#
#			Specifying how much the image size is reduced at each image scale
#
# 			The model has a fixed size defined during training: in the haarcascade_frontalface_alt.xml file !!!
# 			By rescaling the input image, you can resize a larger face to a smaller one,
# 				 making it detectable by the algorithm
#
# 		Value: 1.1 - 1.4
# 			Small -> algorithm will be slow since it is more thorough
# 			High -> faster detection with the risk of missing some faces altogether
#
#  2.) minNeighbors: specifying how many neighbors each candidate rectangle should have to retain it
#  			Value interval: ~ 3-6
#  				Higher values -> less detections but with higher quality !!!
#
#  3.) flags: kind of a heuristic
#  		Reject some image regions that contain too few or too much edges
#  			 and thus can not contain the searched object
#
#  4.) minSize: objects smaller than that are ignored !!!
#  			We can specify what is the smallest object we want to recognize
#  					[30x30] is the standard
#
#  5.) maxSize: objects larger than that are ignored !!!


# Path to image
imagePath = ''

# Path to classifier
cascadeClassifierPath = ''

# Create classifier
cascadeClassifier = cv2.CascadeClassifier(cascadeClassifierPath)
image = cv2.imread(imagePath)
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

detectedFaces = cascadeClassifier.detectMultiScale(grayImage, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))

# Draw rectangles around faces on image
for (x, y, width, height) in detectedFaces:
    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 0, 255), 10)

cv2.imwrite('result.jpg', image)