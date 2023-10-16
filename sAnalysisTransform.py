'''

Sentiment analysis using hugging face transformers 

'''

#import 
from transformers import pipeline 
import pandas as pd 

import numpy as np

#create pipeline for sentiment analysis 
classification = pipeline("sentiment-analysis")

df = pd.read_csv('Datasets/training.1600000.processed.noemoticon.csv',
                 delimiter=',', encoding='ISO-8859-1')
df.columns = ['Sentiment','id','date','query','user','text']
df = df[['Sentiment','text']]


df = df.sample(100)

#positive = 4 , negative = 0

#get the texts into a list (text preprocessing) (model input is commonly lists of strings)
texts = df['text'].to_list()

#print(texts)


#predict (we dont have to train here , just use hugging face pipeline)
predictions = classification(texts)


'''
    the output we get has this form :
        [{'label': 'POSITIVE', 'score': 0.9998},
        {'label': 'NEGATIVE', 'score': 0.9991}]
    
        we notice that both neg and pos have score near 1, 
        we dont want that , we want values between 0 and 1 

        so we will use a simple technique to separate negative and positive predictions 

'''


probas = [d['score'] if d['label'].startswith('P') else 1 - d['score'] for d in predictions]


#do our predictions -> to numerical values 
preds = [1 if d['label'].startswith('P') else 0 for d in predictions]

preds = np.array(preds)

#compute the accuracy of the model 
print("acc:", np.mean(df['target'] == preds))