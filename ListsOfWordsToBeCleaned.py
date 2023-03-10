from NLPConfigFile import*

Contact =["phone","telephone","cellphone","cell","call","landline", "human", "humanbeing", "representative", "agent", ]

Child_tax_credit = ["child", "credit","unenroll","disenroll",""]

Tax_pro_account = ["irs", "need"]

#put lists of words above this function
# makes a list of words that will be removed from the comments
Full_List_Of_Words_To_Be_Removed_From_Comment = []
if Filter_Words_Out_Contact_Config_Variable == 1:
    Full_List_Of_Words_To_Be_Removed_From_Comment.extend(Contact)
if Filter_Words_Out_Child_Tax_Credit_Config_Variable == 1:
    Full_List_Of_Words_To_Be_Removed_From_Comment.extend(Child_tax_credit)
if Filter_Words_Out_Tax_Pro_Account_Config_Variable == 1:
    Full_List_Of_Words_To_Be_Removed_From_Comment.extend(Tax_pro_account)

print(Full_List_Of_Words_To_Be_Removed_From_Comment)
#makes a list of words where if a comment contains one of these words, the comment is considered NaN for the NLP
Full_List_Of_Words_To_Remove_Comments_By = []
if Filter_Comments_Out_Contact_Config_Variable == 1:
    Full_List_Of_Words_To_Remove_Comments_By.extend(Contact)
if Filter_Comments_Out_Child_Tax_Credit_Config_Variable == 1:
    Full_List_Of_Words_To_Remove_Comments_By.extend(Child_tax_credit)
if Filter_Comments_Out_Tax_Pro_Account_Config_Variable == 1:
    Full_List_Of_Words_To_Remove_Comments_By.extend(Tax_pro_account)

print(Full_List_Of_Words_To_Remove_Comments_By)