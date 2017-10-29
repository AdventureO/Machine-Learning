from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

#	Important parameters for SVC: gamma and C
#		gamma -> defines how far the influence of a single training example reaches
#					Low value: influence reaches far
#                   High value: influence reaches close
#
#		C -> trades off hyperplane surface simplicity + training examples missclassifications
#					Low value: simple/smooth hyperplane surface
#					High value: all training examples classified correctly but complex surface

# Load iris data
data_set = datasets.load_iris()

# Get features of data
features = data_set.data
# Get targets of data
targetVariables = data_set.target

# Divide data into four sets (two for fitting and two for testing)
featureTrain, featureTest, targetTrain, targetTest = train_test_split(features, targetVariables, test_size=0.3)

# Create support vector machines model
#model = svm.SVC(gamma=0.001, C=100)
model = svm.SVC()
# Fit model with train data
model.fit(featureTrain, targetTrain)

# Make predictions on test data
predictions = model.predict(featureTest)

# Comparing predictions and true data
# Tell us how many items were classified correctly and how many not
print(confusion_matrix(targetTest, predictions))
# Return percentage of classification accuracy`
print(accuracy_score(targetTest, predictions))