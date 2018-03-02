import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Read data from csv file
creditData = pd.read_csv("credit_data.csv")

# Logistic regression accuracy: ~93%
# KNN accuracy: ~97.5% with transformed features (normalized)
# Naive Bayes accuracy: ~93%

# Choose only three features to make predictions
features = creditData[["income", "age", "loan"]]
# Return column with name default
targetVariables = creditData.default

# Divide data into four sets (two for fitting and two for testing)
featureTrain, featureTest, targetTrain, targetTest = train_test_split(features, targetVariables, test_size=0.3)

# Create naive bayes model
model =GaussianNB()
# fit model with data
model.fit(featureTrain, targetTrain)
# get predictions for test data
predictions = model.predict(featureTest)

# Comparing predictions and true data
# Tell us how many items were classified correctly and how many not
print(confusion_matrix(targetTest, predictions))
# Return percentage of classification accuracy
print(accuracy_score(targetTest, predictions))