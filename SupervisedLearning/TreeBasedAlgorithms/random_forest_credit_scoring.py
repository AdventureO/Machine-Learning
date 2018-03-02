import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

# Logistic regression accuracy: 93%
# we do better with knn: 97.5% !!!!!!!!
# 84% simple kNN without normalizing the dataset
# We can achieve ~ 98.5% with random forests and 25 decision trees

# Read data from csv file
creditData = pd.read_csv("credit_data.csv")

# Choose only three features to make predictions
features = creditData[["income", "age", "loan"]]
# Return column with name default
targetVariables = creditData.default

# Divide data into four sets (two for fitting and two for testing)
featureTrain, featureTest, targetTrain, targetTest = train_test_split(features, targetVariables, test_size=.2)

# Create random forest model and fit it with train data
model = RandomForestClassifier(n_estimators=25).fit(featureTrain, targetTrain)
# Get predictions for test data
predictions = model.predict(featureTest)

# Comparing predictions and true data
# Tell us how many items were classified correctly and how many not
print(confusion_matrix(targetTest, predictions))
# Return percentage of classification accuracy
print(accuracy_score(targetTest, predictions))


