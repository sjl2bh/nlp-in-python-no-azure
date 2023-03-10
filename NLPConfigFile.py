'''This file serves as a configuration file.
Here you can choose the text cleaning that you want applied
as well as other configuration outputs such as if you would like graphs to be outputted on a run.
Most variables will be boolean with a 1 if you want the configuration option to happen
and a 0 if you want the option turned off'''

#FirstDataCleaning variables
    #These variables filter words from comments and leave the remainder of the comment
Filter_Words_Out_Contact_Config_Variable = 0
Filter_Words_Out_Child_Tax_Credit_Config_Variable = 0
Filter_Words_Out_Tax_Pro_Account_Config_Variable = 0

# These variables filter comments containing these words and remove the comment entirely. the comment is not considered in the NLP
Filter_Comments_Out_Contact_Config_Variable = 0
Filter_Comments_Out_Child_Tax_Credit_Config_Variable = 0
Filter_Comments_Out_Tax_Pro_Account_Config_Variable = 0


#SecondExploratoryDataAnalysis variables
Show_Word_Cloud_Plot_config_variable = 1 #determines if the word cloud will be shown. If this is on, you will have to close the plot before the program will continue
Show_Bar_Plot_config_variable = 0 #determines if the bar plot of topics will be shown. If this is on, you will have to close the plot before the program will continue

#ThirdSentimentAnalysis variables
Show_Sentiment_Analysis_Plot_config_variable = 0

#FourthTopicModeling variables
Topics_Count_config_variable = 25 # How many topics would you like your data sorted into


#remove all comments containing nouns
#remove nouns from comments
