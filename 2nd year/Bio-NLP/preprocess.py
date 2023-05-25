# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1PVM6wAtsPZSBSDEXq6B1Ils9cPnh2ncl
"""

from google.colab import drive
drive.mount('/content/drive/')

!pip install transformers

import pandas as pd
from transformers import pipeline
import numpy as np
import re 
#answers from the questionnaire
file = open('/content/drive/MyDrive/BioMedical/chestionar.txt',mode='r')
all_of_it = file.read()
file.close()
questions = all_of_it.split('\n\n')
all_answers = []
for question in questions:
    answers = question.split('\n')
    answers = [x for x in answers]
    all_answers.append(answers)

df_data = []
for data in all_answers:
  question = data[0]
  question_id = question.split('. ')[0]
  question_name = question.split('. ')[1]
  answers = data[1:]
  for ans in answers:
    answer_id = ans.split('. ')[0]
    answer_name = ans.split('. ')[1]
    df_data.append([question_id, question_name, answer_id, answer_name])

df = pd.DataFrame(df_data, columns=['Question Number', 'Question', 'Class', 'Answer'])

df

bdi_data = pd.read_csv('/content/drive/MyDrive/BioMedical/BDI_csv.csv')

bdi_data

from bs4 import BeautifulSoup
import pandas as pd
import re

def preprocessing(text):
    processed_text = text.replace(":)", " happy ")
    processed_text = processed_text.replace("(:", " happy ")
    processed_text = processed_text.replace(":-)", " happy ")
    processed_text = processed_text.replace(":(", " sad ")
    processed_text = processed_text.replace("):", " sad ")
    processed_text = processed_text.replace(":-(", " sad ")

    processed_text = re.sub(r'\b/u\w*\s?', '', processed_text)
    processed_text = re.sub(r'\b/r\w*\s?', '', processed_text)
    processed_text = re.sub(r'http\S+|www\S+|https\S+', '', processed_text)


    popular_special_characters = ['!', '.', ',', '?', '-', '\'']

    # Remove special characters except the most popular ones
    processed_text = re.sub(r"[^\w\s" + re.escape(''.join(popular_special_characters)) + "]", '', text)


    return processed_text

pth_eRisk_train = '/content/drive/MyDrive/BioMedical/train/eRisk_2020_T2_train/Depression Questionnaires_anon.txt'
file = open(pth_eRisk_train, "r")
train = file.readlines()
train = [x.strip() for x in train]

xml_root = '/content/drive/MyDrive/BioMedical/train/eRisk_2020_T2_train/DATA/'


train_data_list = [[] for x in range(21)]
for data in train:
    data = data.split('\t')
    #print(data)
    answers = data[1:]

    print(data)


for data in train:
    data = data.split('\t')
    #print(data)
    answers = data[1:]
    #print(len(answers))

    xml = xml_root + data[0] + '.xml'

    with open(xml, 'r') as f:
        xml_data = f.read()
    
    Bs_data = BeautifulSoup(xml_data, "xml")
    b_title = Bs_data.find_all('TITLE')
    b_text = Bs_data.find_all('TEXT')



    for i in range(len(b_title)):
        title = b_title[i].string
        title = ' ' if title is None else title
        text = b_text[i].string
        text = ' ' if text is None else text
        new_text = title + ' ' + text
        new_text = preprocessing(new_text)


        for no, answer in enumerate(answers):
            #print(no)
            #print(answer)
            v = bdi_data.loc[(bdi_data['Class'] == answer) & (bdi_data['Question Number'] == no+1)].Answer.values[0]
            #print(v)

            pair = [data[0], answer, v, new_text]
            train_data_list[no].append(pair)

for i, train in enumerate(train_data_list):
  ans = bdi_data.loc[(bdi_data['Question Number'] == i+1)].Question.values[0]
  print(ans, i)
  df = pd.DataFrame(train, columns=['Subject', 'Class', 'Answer', 'Post'])
  df.to_csv('/content/drive/MyDrive/BioMedical/data_v1/answer_classes_posts_{}.csv'.format(ans))