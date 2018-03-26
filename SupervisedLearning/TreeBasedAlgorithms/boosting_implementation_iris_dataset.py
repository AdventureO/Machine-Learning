from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

data = datasets.load_iris("path/to/the/dataset")

data.features = data[["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]
data.targets = data.Class

X_train, X_test, y_train, y_test = train_test_split(data.features, data.targets, test_size=0.25, random_state=42)

model = AdaBoostClassifier(n_estimators=100, learning_rate=1, random_state=133)
model.fitted = model.fit(X_train, y_train)
model.predictions = model.fitted.predict(X_test)

print(confusion_matrix(y_test, model.predictions))
print(accuracy_score(y_test, model.predictions))