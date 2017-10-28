from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

# Categories which we will use
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

# Prepare training data from 20newsgroups
trainingData = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

#print("\n".join(trainingData.data[1].split("\n")[:10]))
#print("Target is:", trainingData.target_names[trainingData.target[1]])

# Count the word occurrences
countVectorizer = CountVectorizer()
xTrainCounts = countVectorizer.fit_transform(trainingData.data)
#print(countVectorizer.vocabulary_.get(u'software'))

# Transform the word occurrences into tfidf
tfidTransformer = TfidfTransformer()
xTrainTfidf = tfidTransformer.fit_transform(xTrainCounts)

# Create and fit multinomial naive bayes model
model = MultinomialNB().fit(xTrainTfidf, trainingData.target)

# Create test data
new = ['This has nothing to do with church or religion', 'Software engineering is getting hotter and hotter nowadays']
xNewCounts = countVectorizer.transform(new)
xNewTfidf = tfidTransformer.transform(xNewCounts)

# Predict category for our test data
predicted = model.predict(xNewTfidf)

# Print predictions
for doc, category in zip(new,predicted):
	print('%r --------> %s' % (doc, trainingData.target_names[category]))

