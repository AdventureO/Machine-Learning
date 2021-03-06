from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

# Load iris data
irisData = datasets.load_iris()

# Get features of data
features = irisData.data
# Get targets of data
targetVariables = irisData.target

# Divide data into four sets (two for fitting and two for testing)
featureTrain, featureTest, targetTrain, targetTest = train_test_split(features, targetVariables, test_size=.2)

# Create random forest model and fit it
model = RandomForestClassifier(n_estimators=100).fit(featureTrain, targetTrain)
# Make predictions on test data
predictions = model.predict(featureTest)

# Comparing predictions and true data
# Tell us how many items were classified correctly and how many not
print(confusion_matrix(targetTest, predictions))
# Return percentage of classification accuracy`
print(accuracy_score(targetTest, predictions))
