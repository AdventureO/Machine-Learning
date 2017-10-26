import numpy as np
from sklearn.linear_model import LogisticRegression

# function for this particular example of data
# p i = 1 / 1 + exp[ - ( b0 + b1 * x1 + b2 * x2 + b3 * x3 )]

	#	balance  |        income          |  age     |           class
	#	 10000$		      80 000$            35       1 ( can pay back the debt )
	# 	 7000$		      120 000$           57       1 ( can pay back the debt )
	# 	 100$		      23 000$             22       0 ( can NOT pay back the debt )
	# 	 223$		      18 000$             26       0 ( can NOT pay back the debt )
	# ----------------------------------------------------------------------
	# 	 5500$		      50 000$             25       ? make a prediction

# Create simple dataset
X = np.array([[10000, 80000, 35],[7000, 120000, 57],[100, 23000, 22],[223, 18000, 26]])
Y = np.array([1, 1, 0, 0])

# create logistic regression model
classifier = LogisticRegression()
# fit model with prepared data(3 features)
classifier.fit(X,Y)

print(classifier.predict_proba([5500, 50000, 25]))