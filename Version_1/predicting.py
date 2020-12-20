# all imports
import pandas as pd
from joblib import dump, load

# all imports for learning
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# 1. import data
df = pd.DataFrame(columns=["review_text"], data=[["Wendy's spot is ideal for access to the South Shore and makes drives to both Waimea Canyon and Hanalei manageable. The house is great and gets a consistent breeze throughout. On top of this"]])
df

# 'Wendy's spot is ideal for access to the South Shore and makes drives to both Waimea Canyon and Hanalei manageable. The house is great and gets a consistent breeze throughout. On top of this'
# 'This hotel was so good. It was the best. Very comfortable, kind staff and nice food.'

# 2. Vectorize and transform input data
vect = load('Vectorizer/vectorizer.joblib')
inputVectorized = vect.transform(df)

# 3.1 MultinomialNB load and predict
nb = load('Models/MultinomialNB.joblib')
nb_pred = nb.predict(inputVectorized)
nb_pred 

# 3.2 LogisticRegression load and predict
lr = load('Models/LogisticRegression.joblib')
lr_pred = lr.predict(inputVectorized)
lr_pred

# 3.3 LinearSVC load and predict
lsvc = load('Models/LinearSVC.joblib')
lsvc_pred = lsvc.predict(inputVectorized)
lsvc_pred
