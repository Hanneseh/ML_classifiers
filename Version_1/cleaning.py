# imports
import pandas as pd
from sqlalchemy import create_engine

# reading in the data from Kaggle
dataBooking = pd.read_csv("RawData/Hotel_Reviews.csv", sep=",")

# reading in the hand written reviews
dataHandWritten = pd.read_csv("RawData/Hw_reviews.csv", sep=";")
dataHandWritten

# filtering out the ones that are too short
bookingLongEnoughReviews = (
    dataBooking
        .loc[lambda df: df['Review_Total_Negative_Word_Counts'] >= 5]
        .loc[lambda df: df['Review_Total_Positive_Word_Counts'] >= 5]
)
bookingLongEnoughReviews.describe()

# filtering booking positve reviews, renaming review text and do encoding
positiveReviews = (
    bookingLongEnoughReviews[['Positive_Review']]
    .rename({"Positive_Review" : "review_text"}, axis="columns")
    .assign(sentiment=lambda df: 1)
)

# filtering booking negative reviews, renaming review text and do encoding
negativeReviews = (
    bookingLongEnoughReviews[['Negative_Review']]
    .rename({"Negative_Review" : "review_text"}, axis="columns")
    .assign(sentiment=lambda df: -1)
)

# filtering handwritten positve reviews, renaming review text and do encoding
hwPositiveReviews = (
    dataHandWritten[['positive_reviews']]
    .dropna()
    .rename({"positive_reviews" : "review_text"}, axis="columns")
    .assign(sentiment=lambda df: 1)
)
hwPositiveReviews

# filtering booking negative reviews, renaming review text and do encoding
hwNegativeReviews = (
    dataHandWritten[['negative_review']]
    .dropna()
    .rename({"negative_review" : "review_text"}, axis="columns")
    .assign(sentiment=lambda df: -1)
)
hwNegativeReviews

# combining all data frames to one
allReviews = (
    positiveReviews
    .append(hwPositiveReviews)
    .append(negativeReviews)
    .append(hwNegativeReviews)
)
allReviews
allReviews.dtypes

# shuffeling and resetting index
shuffledReviews = allReviews.sample(frac=1).reset_index(drop=True)
shuffledReviews.describe()
shuffledReviews

# storing the clean data in a database
engine = create_engine('mysql+mysqlconnector://test:root@localhost/cleaned_hotel_reviews')
shuffledReviews.to_sql(name='all_reviews',con=engine,if_exists='fail',index=False, chunksize=1000) 
