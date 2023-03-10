'''This notebook goes through a necessary step of any data science project - data cleaning. Data cleaning is a time consuming and unenjoyable task, yet it's a very important one. Keep in mind, "garbage in, garbage out". Feeding dirty data into a model will give us results that are meaningless.

Specifically, we'll be walking through:

*Getting the data - *in this case, we'll be scraping data from a website
*Cleaning the data - *we will walk through popular text pre-processing techniques
*Organizing the data - *we will organize the cleaned data into a way that is easy to input into other algorithms
The output of this notebook will be clean, organized data in two standard text formats:

Corpus - a collection of text
Document-Term Matrix - word counts in matrix format

Problem Statement: As a reminder, our goal is to look at IRS survey responses. Specifically, we'd like to know about sentiment of responses and the common topics that are brought up by respondents.

Our Data comes from excel downloads off of foresee survey data. Before running this program, the excel file should be
in the format of: Column A is respondent IDs and Column B is the Answers to your survey question.'''

# Web scraping, pickle imports
import requests  # must install in conda
from bs4 import BeautifulSoup
import pickle
import pandas as pd  # must install openpyxl in conda
# Apply a first round of text cleaning techniques
import re
import string
from sklearn.feature_extraction.text import CountVectorizer  # Must install in conda using "conda install -c conda-forge scikit-learn"
from NLPConfigFile import*
from ListsOfWordsToBeCleaned import*

# These make the full pandas dataframe available to view in the output window of python instead of the default truncated version
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

file_name = 'C:\\Users\\617626\\Desktop\\IRS\\NLP survey project\\Non-Code\\Survey_ID_and_Responses.xlsx'
xl_workbook = pd.ExcelFile(file_name)  # Load the excel workbook
input_df = xl_workbook.parse("DataSheet")  # Parse the sheet into a dataframe
print(input_df)
List_Of_Respondent_IDs = input_df.iloc[:, 0].tolist()  # Cast column A into a python list, will skip row 1
List_Of_Responses = input_df.iloc[:, 1].tolist()  # Cast column B into a python list, will skip row 1

# Check if data came in correctly
# for i in List_Of_Respondent_IDs:
#     print(List_Of_Respondent_IDs)
#
# for i in List_Of_Responses:
#     print(List_Of_Responses)
# print(len(List_Of_Respondent_IDs))
# print(len(List_Of_Responses))
# print(List_Of_Respondent_IDs[0])

# Changing lists to dictionary using zip() ID; response
Dictionary_Of_Data = dict(zip(List_Of_Respondent_IDs, List_Of_Responses))
# print ("Resultant dictionary is : " +  str(res))

# Puts Dictionary into pandas dataframe
data_df = pd.DataFrame.from_dict([Dictionary_Of_Data]).transpose()
data_df.columns = ['RESPONSES']

#data_df = data_df.sort_index() ##################Commented this out cause I think it is unnecessary

# Check for first 5 lines of dataframe
# print(data_df.head())
# print(data_df.dtypes)

########################################################
# TEXT CLEANING
# Apply a first round of cleaning (must always be done)
def clean_text_round1(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = str(text)
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
#    text = re.sub('\w*\d\w*', '', text)
    return text
round1 = lambda x: clean_text_round1(x)
# Cleans the text with round one regular expressions
data_clean = pd.DataFrame(data_df.RESPONSES.apply(round1))

# print(data_clean)

# Apply a second round of cleaning (must always be done)
def clean_text_round2(text):
    '''Get rid of some additional punctuation and non-sensical text that was missed the first time around.'''
    text = str(text)
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', '', text)
    return text
round2 = lambda x: clean_text_round2(x)
data_clean = pd.DataFrame(data_clean.RESPONSES.apply(round2))

################## Optional text cleaning
def remove_words_from_comments(text):
    '''If a comment contains a word specified by a list/set of list
    in the parameters, the word is removed and the rest of the comment
    is kept'''
    text = str(text)
    for Word_To_Be_Removed in range(0, len(Full_List_Of_Words_To_Be_Removed_From_Comment)):
        text = re.sub(Full_List_Of_Words_To_Be_Removed_From_Comment[Word_To_Be_Removed],'',text)
    return text
Remove_Words = lambda x: remove_words_from_comments(x)
data_clean = pd.DataFrame(data_clean.RESPONSES.apply(Remove_Words))

def remove_comments_containing_filtered_words(text):
    '''If a comment contains a word specified by a list/set of list
    in the parameters, the word is removed and the rest of the comment
    is kept'''
    text = str(text)
    for Word_To_Remove_Comment_By in range(0, len(Full_List_Of_Words_To_Remove_Comments_By)):
        if Full_List_Of_Words_To_Remove_Comments_By[Word_To_Remove_Comment_By] in text:
            text = ""
    return text
Remove_Comments = lambda x: remove_comments_containing_filtered_words(x)
data_clean = pd.DataFrame(data_clean.RESPONSES.apply(Remove_Comments))



def remove_comments_containing_nan(text):
    '''THIS MUST BE AT THE END OF TEXT CLEANING. If a comment is just "nan" in the pandas dataframe (often happens in text cleaning), then it will be replaced with a blank space so that the
    topic modelling is not affected by the nonsense word "nan"'''
    text = str(text)
    if text == "nan":
        text = ""
    return text
Remove_nan_Comments = lambda x: remove_comments_containing_nan(x)
data_clean = pd.DataFrame(data_clean.RESPONSES.apply(Remove_nan_Comments))
#print(data_clean)
#########THIS LINE COMPLETES THE FINAL CORPUS

# We are going to create a document-term matrix using CountVectorizer, and exclude common English stop words
cv = CountVectorizer(stop_words='english')
data_cv = cv.fit_transform(data_clean.RESPONSES)
data_dtm_temporary = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_dtm_temporary.index = data_clean.index
data_dtm = data_dtm_temporary  # this line is necessary for the variable to be imported properly to other files
#print(data_dtm)
# This will take a long time
# data_dtm.to_excel("C:\\Users\\617626\\Desktop\\IRS\\NLP survey project\\Non-Code\\CheckFileForProject.xlsx")

# All Data is now cleared of numbers, words that contain numbers, nonsenseical text, stop words, punctuation, brackets and we have both a corpus and a document term matrix
# acronymns like ctc, irs, ect. have been left in for topic modelling

print("We did it")