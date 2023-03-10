'''
MOST IMPORTANT TO JEFF
WANT TO APPLY SENTIMENTS TO THOSE TOPICS (SEE WHAT TOPICS ARE TRENDING NEGATIVE AND POSITIVE
WANT AN IDEA OF WHAT COMPLAINTS ARE ABOUT MORE IN DEPTH THAN TOPIC MODELING
    ex: WHY ARE PEOPLE SAYING NEGATIVE THINGS ON CTC? IS IT BECAUSE THEY CANT FIND IT ONLINE, THE UI IS BAD, EST.
    MAYBE TRY TOPIC MODELING AT A SECOND OR THIRD LEVEL
    DO TOPIC MODELING AGAIN ON ONLY COMMENTS RELATED TO CTC
THERE IS A TEAM AT VETERANS AFFAIRS WE MAY WANT TO MEET WITH THAT DOES NLP ON SURVEY DATA
'''

import pandas as pd
from wordcloud import WordCloud  # Will need to install using   conda install -c conda-forge wordcloud
import matplotlib.pyplot as plt
from textblob import TextBlob  # conda install -c conda-forge textblob
import numpy as np
import math
from sklearn.feature_extraction.text import CountVectorizer
from gensim import matutils, models, corpora
from gensim.utils import simple_preprocess
import scipy.sparse
import nltk #you will have to use pip install to install this in conda
from nltk import word_tokenize, pos_tag
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
import pyLDAvis.gensim # you will have to use this statement in conda -> "pip install pyLDAvis==2.1.1"
from FirstDataCleaning import input_df, data_dtm, data_clean
from ThirdSentimentAnalysis import data_sentiment
from NLPConfigFile import*



data = data_dtm
data_clean = data_clean
# One of the required inputs is a term-document matrix
tdm = data.transpose()
print(tdm.head())

sparse_counts = scipy.sparse.csr_matrix(tdm)
corpus = matutils.Sparse2Corpus(sparse_counts)

# Gensim also requires dictionary of the all terms and their respective location in the term-document matrix
# the next few lines are reused
cv = CountVectorizer(stop_words='english')
data_cv = cv.fit_transform(data_clean.RESPONSES)
data_dtm_temporary = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_dtm_temporary.index = data_clean.index
# print(data_dtm)
id2word = dict((v, k) for k, v in cv.vocabulary_.items())

''' Now that we have the corpus (term-document matrix) and id2word (dictionary of location: term),
# we need to specify two other parameters as well - the number of topics and the number of passes'''
#lda = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=2, passes=10)
''' You will likely need to change the number of topics and passes
 for each survey set
 you will likely need to remove more words from your data'''
#print(lda.print_topics())
#Maybe want this to print to a word doc or something

#lda = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=20, passes=80)
#print(lda.print_topics())
'''Maybe try more or less topics
Maybe try removing the common words like notice, letter, payment, ect.
Maybe try adding back words with numbers (like form 1040A, est.) (Tried this and it helped a lot)
Maybe filter the repository on nouns and adjectives'''
def nouns(text):
    '''Given a string of text, tokenize the text and pull out only the nouns.'''
    is_noun = lambda pos: pos[:2] == 'NN'
    tokenized = word_tokenize(text)
    all_nouns = [word for (word, pos) in pos_tag(tokenized) if is_noun(pos)]
    return ' '.join(all_nouns)

'''Come back to the Nouns part, I am not sure what it is doing. I think it is only allowing the topic modelling to be 
based on nouns, but the keywords identified for topics can be any word. I also don't know if this needs
a lambda in front of it or if I should maybe toss this altogether'''
data_nouns = pd.DataFrame(data_clean.RESPONSES.apply(nouns))
data_nouns

# Recreate a document-term matrix with only nouns
cvn = CountVectorizer(stop_words='english')
data_cvn = cvn.fit_transform(data_nouns.RESPONSES)
data_dtmn = pd.DataFrame(data_cvn.toarray(), columns=cvn.get_feature_names())
data_dtmn.index = data_nouns.index
#print(data_dtmn)
id2word = dict((z, q) for q, z in cvn.vocabulary_.items())



tdm = data_dtmn.transpose()
sparse_counts = scipy.sparse.csr_matrix(tdm)
corpus = matutils.Sparse2Corpus(sparse_counts)

#line that model is actually made
# lda = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=Topics_Count_config_variable, passes=80) #original line
lda = models.LdaModel(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=10,
                                       passes=20)
# for i in range(Topics_Count_config_variable):
# #     print(lda.print_topic(i))
# #     print("\n")

'''I would like to add pyldavis visualization to the model, but
it is really hard. kept getting dict object has no attribute token2id.
maybe it can be fixed by loading in data and making it the
way people do in examples https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/'''
# lda_display = pyLDAvis.gensim.prepare(lda, corpus1, id2word1)
# pyLDAvis.display(lda_display)



def format_topics_sentences(lda=lda, corpus=corpus, texts=data):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(lda[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = lda.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)

data = data_clean.RESPONSES.tolist()
print(data_clean.head(10))

df_topic_sents_keywords = format_topics_sentences(lda=lda, corpus=corpus, texts=data)
print(df_topic_sents_keywords.head(10))


# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Cleaned Comment']

print(df_dominant_topic.head(10))

#join all dataframes together by column for final results
data_sentiment = data_sentiment.drop(data_sentiment.columns[[0]], axis=1) #drop the responses in data_sentiment
print(data_sentiment)
df_temp_output = pd.concat([d.reset_index(drop=True) for d in [df_dominant_topic, data_sentiment]], axis=1) #put df_dominant_topic and data_sentiment side by side
#input_df = input_df.drop(input_df.columns[[0,1]], axis=1) #drop respondent id and
df_final_output = pd.concat([d.reset_index(drop=True) for d in [df_temp_output, input_df]], axis=1)

print(df_final_output.head(10))

# Show
#print(df_dominant_topic.head(10))
df_final_output.to_excel("C:\\Users\\617626\\Desktop\\IRS\\NLP survey project\\Non-Code\\TopicModeledComments.xlsx")
#NOW I WANT TO PRINT THIS TO EXCEL AND RUN INDIVIDUAL TOPICS AGAIN

