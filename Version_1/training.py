# all imports for handling data
import pandas as pd
from sqlalchemy import create_engine
import mysql.connector
from joblib import dump, load

# all imports for learning
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, precision_score, recall_score

# 1. retrive data via a stored procedure and populate dataframe
connection = mysql.connector.connect(host='localhost', database='cleaned_hotel_reviews', user='test', password='root')
cursor = connection.cursor()
procedureCall = cursor.callproc('getReviewsByMaxSentimentValue', [1]) 
for i in cursor.stored_results(): results = i.fetchall()
df = pd.DataFrame(results, columns=['review_text', 'lable'])
cursor.close()
connection.close()

# checking the data
df
df.dtypes # because lable needs to be an int
df.lable.value_counts() # checking the balance

# 2. splitting into training and testing. 
X = df["review_text"]
y = df["lable"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 3.1 Vectorize and transform the data for the model
vect = CountVectorizer(
    stop_words='english', 
    binary=True, 
    ngram_range=(1,2)
    )

X_train_vect = vect.fit_transform(X_train)
X_test_vect = vect.transform(X_test)

# 3.2 save vectorizer on disk for later use
dump(vect, 'Vectorizer/vectorizer.joblib')

# 4.1 Train and evaluate Classifier MultinomialNB
nb = MultinomialNB()
nb.fit(X_train_vect, y_train)

nb_y_pred = nb.predict(X_test_vect)

nb_acc = accuracy_score(y_test, nb_y_pred)
nb_cm = confusion_matrix(y_test, nb_y_pred)
nb_f1 = f1_score(y_test, nb_y_pred)
nb_precision = precision_score(y_test, nb_y_pred)
nb_recall = recall_score(y_test, nb_y_pred)
nb_auc = roc_auc_score(y_test, nb_y_pred)

# 4.2 Train and evaluate Classifier LogisticRegression
lr = LogisticRegression()
lr.fit(X_train_vect, y_train)

lr_y_pred = lr.predict(X_test_vect)

lr_acc = accuracy_score(y_test, lr_y_pred)
lr_cm = confusion_matrix(y_test, lr_y_pred)
lr_f1 = f1_score(y_test, lr_y_pred)
lr_precision = precision_score(y_test, lr_y_pred)
lr_recall = recall_score(y_test, lr_y_pred)
lr_auc = roc_auc_score(y_test, lr_y_pred)

# 4.3 Train and evaluate Classifier LinearSVC
lsvc = LinearSVC()
lsvc.fit(X_train_vect, y_train)

lsvc_y_pred = lsvc.predict(X_test_vect)

lsvc_acc = accuracy_score(y_test, lsvc_y_pred)
lsvc_cm = confusion_matrix(y_test, lsvc_y_pred)
lsvc_f1 = f1_score(y_test, lsvc_y_pred)
lsvc_precision = precision_score(y_test, lsvc_y_pred)
lsvc_recall = recall_score(y_test, lsvc_y_pred)
lsvc_auc = roc_auc_score(y_test, lsvc_y_pred)

# 5. Creating tables to present the metrics
metricsFormatted = [
    (
        'MultinomialNB', 
        "{:.2f}%".format(nb_acc * 100), 
        "{:.2f}%".format(nb_f1 * 100), 
        "{:.2f}%".format(nb_precision * 100), 
        "{:.2f}%".format(nb_recall * 100), 
        "{:.4f}".format(nb_auc), nb_cm),
    (
        'LogisticRegression', 
        "{:.2f}%".format(lr_acc * 100), 
        "{:.2f}%".format(lr_f1 * 100), 
        "{:.2f}%".format(lr_precision * 100), 
        "{:.2f}%".format(lr_recall * 100), 
        "{:.4f}".format(lr_auc), lr_cm),
    (
        'LinearSVC', 
        "{:.2f}%".format(lsvc_acc * 100), 
        "{:.2f}%".format(lsvc_f1 * 100), 
        "{:.2f}%".format(lsvc_precision * 100), 
        "{:.2f}%".format(lsvc_recall * 100), 
        "{:.4f}".format(lsvc_auc), lsvc_cm),
    ]

metricsForPlotting = [
    ('MultinomialNB', nb_acc, nb_f1, nb_precision, nb_recall, nb_auc, nb_cm),
    ('LogisticRegression', lr_acc, lr_f1, lr_precision, lr_recall, lr_auc, lr_cm),
    ('LinearSVC', lsvc_acc, lsvc_f1, lsvc_precision, lsvc_recall, lsvc_auc, lsvc_cm),
    ]

metricsFormattedDF = pd.DataFrame(
    metricsFormatted, columns = ['Classifier', 'Accuracy', 'F1', 'Precision', 'Recall', 'AUC', 'CM'])
metricsDF = pd.DataFrame(
    metricsForPlotting, columns = ['Classifier', 'Accuracy', 'F1', 'Precision', 'Recall', 'AUC', 'CM'])

# 6. making all data persistent
dump(nb, 'Models/MultinomialNB.joblib')
dump(lr, 'Models/LogisticRegression.joblib')
dump(lsvc, 'Models/LinearSVC.joblib')

dump(metricsDF, 'Metrics/metrics.joblib')
dump(metricsFormattedDF, 'Metrics/metricsFormatted.joblib')