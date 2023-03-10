'''
This file does sentiment analysis on each response and rates each response on how negatively/positively emotive and how factual/opinionated it is
The end result of this file is a scatterplot of polarity vs. subjectivity
'''
import pandas as pd
from wordcloud import WordCloud  # Will need to install using   conda install -c conda-forge wordcloud
import matplotlib.pyplot as plt
from textblob import TextBlob  # conda install -c conda-forge textblob
import numpy as np
import math
from FirstDataCleaning import data_dtm, data_clean
from NLPConfigFile import*

data_sentiment = data_clean  # variable "data" will be our corpus of type pandas.core.frame.DataFrame

# Create quick lambda functions to find the polarity and subjectivity of each response
pol = lambda x: TextBlob(x).sentiment.polarity
sub = lambda x: TextBlob(x).sentiment.subjectivity

data_sentiment['polarity'] = data_sentiment['RESPONSES'].apply(pol)
data_sentiment['subjectivity'] = data_sentiment['RESPONSES'].apply(sub)

plt.rcParams['figure.figsize'] = [10, 8]

for index, respondent in enumerate(data_sentiment.index):
    x = data_sentiment.polarity.loc[respondent]
    y = data_sentiment.subjectivity.loc[respondent]
    plt.scatter(x, y, color='blue')

plt.title('Sentiment Analysis', fontsize=20)
plt.xlabel('<-- Negative -------- Positive -->', fontsize=15)
plt.ylabel('<-- Facts -------- Opinions -->', fontsize=15)
if Show_Sentiment_Analysis_Plot_config_variable == 1:
    plt.show()
print(data_sentiment)

#may want to track sentiments compared with other columns of foresee data, for instance the sentiment based on reason for visit, or time of response, or est.