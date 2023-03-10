import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
import spacy
import en_core_web_md
'''you will have to import spacy in anaconda prompt: 
conda install -c conda-forge spacy
python -m spacy download en_core_web_sm
'''
from pprint import pprint
from gensim.models import CoherenceModel
import pyLDAvis.gensim
import pickle
import pyLDAvis
'''you will need have to import ipython in anaconda:
 conda install -c anaconda ipython
'''
import numpy as np
import tqdm
import pandas as pd
from IPython.core.display import HTML


from NLPConfigFile import*
from FirstDataCleaning import input_df, data_clean
from ThirdSentimentAnalysis import data_sentiment


data_clean = data_clean #corpus
# print(data_clean.columns)
# print(data_clean.head())

# Let’s tokenize each sentence into a list of words, removing punctuations and unnecessary characters altogether.
def sent_to_words(comments):
    for comment in comments:
        yield(gensim.utils.simple_preprocess(str(comment), deacc=True))  # deacc=True removes punctuations
'''gensim simple_preprocess is getting rid of numbers and can be replaced by another tokenizer'''

data = data_clean.RESPONSES.values.tolist()
data_words = list(sent_to_words(data))


# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)


# Define functions for bigrams, trigrams and lemmatization
def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

# Form Bigrams
data_words_bigrams = make_bigrams(data_words)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(data_lemmatized)

#The two main inputs to the LDA topic model are the dictionary(id2word) and the corpus. Let’s create them.
# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
print(corpus) #this corpus is wierd and is a list of numbers. works for the model but be cautious when using

# Build LDA model
'''runtime can likely be improved with gensim.models.LdaMulticore'''
lda_model = lda_model = gensim.models.LdaModel(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=Topics_Count_config_variable,
                                       random_state=100,
                                       chunksize=100,
                                       passes=20,
                                       per_word_topics=True)
# Print the Keyword in the topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]


# Compute Coherence Score
'''can likely be improved by setting processes above 1 and using multicore processing'''
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v', processes=1)
coherence_lda = coherence_model_lda.get_coherence()
print('Coherence Score: ', coherence_lda)

#########
'''BEGIN HYPERPARAM TUNING'''
# Model hyperparameters can be thought of as settings for a machine learning algorithm that are tuned by the data scientist before training. Examples would be the number of trees in the random forest, or in our case, number of topics K
# Model parameters can be thought of as what the model learns during training, such as the weights for each word in a given topic.
# Number of Topics (K)
# Dirichlet hyperparameter alpha: Document-Topic Density
# Dirichlet hyperparameter beta: Word-Topic Density

# supporting function
def compute_coherence_values(corpus, dictionary, k, a, b):
    lda_model = gensim.models.LdaModel(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=k,
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           alpha=a,
                                           eta=b)

    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v', processes=1)

    return coherence_model_lda.get_coherence()


grid = {}
grid['Validation_Set'] = {}

# Topics range
min_topics = 2
max_topics = 20 #set back to 11
step_size = 1
topics_range = range(min_topics, max_topics, step_size)

# Alpha parameter
alpha = list(np.arange(0.01, 1, 0.3))
alpha.append('symmetric')
alpha.append('asymmetric')

# Beta parameter
beta = list(np.arange(0.01, 1, 0.3))
beta.append('symmetric')

# Validation sets
num_of_docs = len(corpus)
corpus_sets = [gensim.utils.ClippedCorpus(corpus, int(num_of_docs*0.75)),
               corpus]

corpus_title = ['75% Corpus', '100% Corpus']

model_results = {'Validation_Set': [],
                 'Topics': [],
                 'Alpha': [],
                 'Beta': [],
                 'Coherence': []
                }

# Can take a long time to run
# if 1 == 1:
#     pbar = tqdm.tqdm(total=(len(beta) * len(alpha) * len(topics_range) * len(corpus_title)))
#
# #    iterate through validation corpuses
#     for i in range(len(corpus_sets)):
#         # iterate through number of topics
#         for k in topics_range:
#             # iterate through alpha values
#             for a in alpha:
#                 # iterate through beta values
#                 for b in beta:
#                     # get the coherence score for the given parameters. Can likely be improved with multithreading
#                     cv = compute_coherence_values(corpus=corpus_sets[i], dictionary=id2word,
#                                                   k=k, a=a, b=b)
#                     # Save the model results
#                     model_results['Validation_Set'].append(corpus_title[i])
#                     model_results['Topics'].append(k)
#                     model_results['Alpha'].append(a)
#                     model_results['Beta'].append(b)
#                     model_results['Coherence'].append(cv)
#
#                     pbar.update(1)
#     pd.DataFrame(model_results).to_csv('C:\\Users\\617626\\Desktop\\IRS\\NLP survey project\\Non-Code\\LDA_tuning_results.csv', index=False)
#     pbar.close()
########

#
#
#
#

visualisation = pyLDAvis.gensim.prepare(lda_model, corpus, id2word, mds='mmds')
pyLDAvis.save_html(visualisation, 'C:\\Users\\617626\\Desktop\\IRS\\NLP survey project\\Non-Code\\LDA_Visualization.html')
#yLDAvis.display(HTML('C:\\Users\\617626\\Desktop\\IRS\\NLP survey project\\Non-Code\\LDA_Visualization.html'))
#pyLDAvis.show(visualisation)



print("we got here")
#
#
######## making excel report (this takes a long time. May be room for improvement by using better algorithms and/or different data objects

def format_topics_sentences(ldamodel=None, corpus=corpus, texts=data):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
print(df_dominant_topic.head(100))

data_sentiment = data_sentiment.drop(data_sentiment.columns[[0]], axis=1) #drop the responses in data_sentiment
print(data_sentiment)
df_temp_output = pd.concat([d.reset_index(drop=True) for d in [df_dominant_topic, data_sentiment]], axis=1) #put df_dominant_topic and data_sentiment side by side
#input_df = input_df.drop(input_df.columns[[0,1]], axis=1) #drop respondent id and
df_final_output = pd.concat([d.reset_index(drop=True) for d in [df_temp_output, input_df]], axis=1)

print(df_final_output.head(10))

# Show
#print(df_dominant_topic.head(10))
df_final_output.to_excel("C:\\Users\\617626\\Desktop\\IRS\\NLP survey project\\Non-Code\\TopicModeledComments.xlsx")