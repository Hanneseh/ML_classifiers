# https://towardsdatascience.com/sentiment-analysis-with-pyspark-bc8e83f80c35

# imports
import findspark
findspark.init()
import pyspark as ps
import warnings
from pymongo import MongoClient
import pandas as pd
from pyspark.sql import SQLContext
from pyspark.ml.feature import IDF, Tokenizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml import PipelineModel
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import NGram, VectorAssembler


# getting the data from mongoDB
client = MongoClient('localhost', 27017)
db = client['individualAssignment']
balancedReviews = db.balancedReviews
hotelReviews = pd.DataFrame(list(balancedReviews.find({},{"_id" : 0})))

# setting up the local spark "cluster"
try:
    sc = ps.SparkContext('local[4]') # 4 means its using 4 cores. If you would like to use less or more change this number accordingliy
    sqlContext = SQLContext(sc) # data structure for cluster
except ValueError:
    warnings.warn("SparkContext already exists in this scope")

# creating a spark data frame with the input of a pandas dataframe
df = sqlContext.createDataFrame(hotelReviews)
df.show(5)
df.count()

# splitting into training and testing data
(train_set, test_set) = df.randomSplit([0.80, 0.20], seed = 2000)


################# this function returns a pipeline and is called later ##########

# this function returns a pipeline which is doing many things at once. read further comments:
def build_ngrams_wocs(inputCol=["review_text","sentiment"], n=3):
    # initiating tokenizer
    tokenizer = [Tokenizer(inputCol="review_text", outputCol="words")]
    # defining ngram range
    ngrams = [
        NGram(n=i, inputCol="words", outputCol="{0}_grams".format(i))
        for i in range(1, n + 1)
    ]
    # defining count vectorizer
    cv = [
        CountVectorizer(vocabSize=5460,inputCol="{0}_grams".format(i),
            outputCol="{0}_tf".format(i))
        for i in range(1, n + 1)
    ]
    # defining term frequency 
    idf = [IDF(inputCol="{0}_tf".format(i), outputCol="{0}_tfidf".format(i), minDocFreq=5) for i in range(1, n + 1)]
    # assembles vectors and idf to features
    assembler = [VectorAssembler(
        inputCols=["{0}_tfidf".format(i) for i in range(1, n + 1)],
        outputCol="features"
    )]
    # setting labels for training
    label_stringIdx = [StringIndexer(inputCol = "sentiment", outputCol = "label")]
    # initializing the classifier (this could also be another classifier like random forrest)
    lr = [LogisticRegression(maxIter=100)]
    # returns final pipeline
    return Pipeline(stages=tokenizer + ngrams + cv + idf+ assembler + label_stringIdx+lr)


################ training and testing of the model #######################
%%time
trigramwocs_pipelineFit = build_ngrams_wocs().fit(train_set)  # training
predictions_wocs = trigramwocs_pipelineFit.transform(test_set) # testing
accuracy_wocs = predictions_wocs.filter(predictions_wocs.label == predictions_wocs.prediction).count() / float(test_set.count()) # calculating accuracy

# setting up the evaluator for evaluationg and predicting later
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")

roc_auc_wocs = evaluator.evaluate(predictions_wocs) # calculating roc_auc


# print accuracy, roc_auc
print("Accuracy Score: {0:.4f}".format(accuracy_wocs))
print("ROC-AUC: {0:.4f}".format(roc_auc_wocs))


#################### play around with the model ##############################
# write fictional review
newReview = pd.DataFrame({'review_text':['I was fine with the hotel, nothing special']})

# transfrom review to compatiple data type
dfSparkReview = sqlContext.createDataFrame(newReview)

# predict review
predictions_wocs = trigramwocs_pipelineFit.transform(dfSparkReview)

# display prediction output
predictions_wocs.head()

# display class the vector belongs to
predictions_wocs.select('prediction').show()

# Now we can save the fitted pipeline to disk
trigramwocs_pipelineFit.save('SerializedModels/SparkLogistigRegressionModel')

##################################################
# retrieve pipeline again if you want to use it without training

# befor running this line set up the cluster, otherwise it doesnt work
myPipeline =  PipelineModel.load('SerializedModels/SparkLogistigRegressionModel')


#################### play around with the model ##############################
# write fictional review
newReview = pd.DataFrame({'review_text':['Nice good, very good']})

# transfrom review to compatiple data type
dfSparkReview = sqlContext.createDataFrame(newReview)

# predict review
predictions_wocs = myPipeline.transform(dfSparkReview)

# display prediction output
predictions_wocs.head()

# display class the vector belongs to
predictions_wocs.select('prediction').show()

# Now we can save the fitted pipeline to disk
trigramwocs_pipelineFit.save('SerializedModels/SparkLogistigRegressionModel')