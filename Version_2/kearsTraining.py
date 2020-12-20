# https://machinelearningmastery.com/predict-sentiment-movie-reviews-using-deep-learning/

# all imports
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense

from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from pymongo import MongoClient
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from numpy import asarray
from numpy import zeros
from matplotlib import pyplot as plt
from joblib import dump, load


# database connection
client = MongoClient('localhost', 27017)
db = client['individualAssignment']
balancedReviews = db.balancedReviews

# loading data from mongoDB in python
hotelReviews = pd.DataFrame(list(balancedReviews.find({},{"_id" : 0})))

# splitting into x and y (date and label)
X = []
sentences = list(hotelReviews['review_text'])
for sen in sentences:
    X.append(sen)

y = []
sentiment = list(hotelReviews['sentiment'])
for sen in sentiment:
    y.append(sen)

#  splitting into test and train sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# initializing tokenizer
max_words=5000
tokenizer = Tokenizer(num_words=max_words)

# fitting tokenizer
tokenizer.fit_on_texts(X_train)

# sequenzing reviews, top words first
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

# defining vocab size
vocab_size = len(tokenizer.word_index) + 1

# padding reviews to fit input layer of model
max_len = 500
X_train = pad_sequences(X_train, padding='post', maxlen=max_len)
X_test = pad_sequences(X_test, padding='post', maxlen=max_len)

# importing data for embedding layer
embeddings_dictionary = dict()
glove_file = open('glove.6B.100d.txt', encoding="utf8")

# populating embedding_dict with words
for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary [word] = vector_dimensions
glove_file.close()

# setting up embedding matrix
embedding_matrix = zeros((vocab_size, 100))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

# displaying shape of embedding matrix
embedding_matrix.shape

# defining and compiling the model with all parameters
model = Sequential()
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=max_len , trainable=False)
model.add(embedding_layer)
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

# look at model set up
print(model.summary())

# Fit/Train the model (takes time!)
%%time
history = model.fit(X_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=1)

# Print evaluation scores
print("Test Score:", scores[0])
print("Test Accuracy:", scores[1])

# visualization of model learning progress over each epoch
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()


################################# play around with the model ################

# type some review in here
#instance=["I hate it, it was so dirty and noisy"]
#instance=["What a nice hotel. I felt very welcome"]
instance=["I was fine with the hotel, nothing special"]

# tokenize the review
instance = myTokenzier.texts_to_sequences(instance)

# flatten the tokenized review for the model
flat_list = []
for sublist in instance:
    for item in sublist:
        flat_list.append(item)
flat_list = [flat_list]

# pad the review to fit the input layer
instance = pad_sequences(flat_list, padding='post', maxlen=max_len)

# predict the sentiment of the review (this line prints out the class the review belongs to as a range between 0 to 1)
myTrainedModel.predict(instance)

# storing the model and tokenizer on disk if you want to use it later again and dont want to go through the training
dump(model, 'SerializedModels/kerasModel.joblib')
dump(tokenizer, 'SerializedModels/KerasTokenizer.joblib')

# load the model to use is again without having to train it
myTrainedModel = load('SerializedModels/kerasModel.joblib')
myTokenzier = load('SerializedModels/KerasTokenizer.joblib')