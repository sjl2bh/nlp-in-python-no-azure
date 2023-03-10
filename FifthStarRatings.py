'''
This file will use sentiment analysis to give a 1-5 satisfaction rating to each comment
The machine rating will then be compared to the original survey results to see
how survey written responses compare to multiple choice answers
'''

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from FirstDataCleaning import data_dtm, data_clean
from NLPConfigFile import*



data = data_clean #data is our corpus

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")


#Function reads a review and returns a 1-5 rating of customer satisfaction
def sentiment_score(RESPONSES):
    tokens = tokenizer.encode(RESPONSES, return_tensors='pt')
    result = model(tokens)
    return int(torch.argmax(result.logits))+1


print(sentiment_score(data['RESPONSES'].iloc[1]))

plt.rcParams['figure.figsize'] = [10, 8]

# for index, respondent in enumerate(data.index):
#     x = data.polarity.loc[respondent]
#     y = data.subjectivity.loc[respondent]
#     plt.scatter(x, y, color='blue')
#
#
# # Gives a graph of machine rating vs. respondent rating
# plt.title('Sentiment Analysis', fontsize=20)
# plt.xlabel('<-- Negative -------- Positive -->', fontsize=15)
# plt.ylabel('<-- Facts -------- Opinions -->', fontsize=15)
#
# plt.show()
