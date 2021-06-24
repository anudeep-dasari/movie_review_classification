# -*- coding: utf-8 -*-

import os
import pandas as pd

train_reviews=[]
test_reviews=[]
# Function to load data from directory
def load_data (data_directory):
    for tlabel in ['train','test']:
        for label in ['pos','neg']:
            directory = f"{data_directory}\{tlabel}\{label}"
            for review in os.listdir(directory):
                if review.endswith('.txt'):
                    with open(f"{directory}\{review}", encoding="utf8") as f:
                        text = f.read()
                        if tlabel == 'train':
                            train_reviews.append(text)
                        else :
                            test_reviews.append(text)
                        

#Run the function for current directory   
load_data('aclimdb')


#Convert the list to dataframe
train_data = pd.DataFrame(train_reviews)
test_data = pd.DataFrame(test_reviews)


#Add Sentiment column
s = train_data.index<12500
train_data['Sentiment'] = s.astype(int)
#train_data = train_data.drop('index', axis=1)
train_data.columns = ['Reviews','Sentiment']

s = test_data.index<12500
test_data['Sentiment'] = s.astype(int)
#test_data = test_data.drop('index', axis=1)
test_data.columns = ['Reviews','Sentiment']


# Write data to csv
train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)






