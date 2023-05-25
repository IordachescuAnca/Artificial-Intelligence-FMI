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

pth_eRisk_train = '/home/anca/Fac/bio_nlp/dataset/train/train_dataset/Depression Questionnaires_anon.txt'
file = open(pth_eRisk_train, "r")
train = file.readlines()
train = [x.strip() for x in train]

xml_root = '/home/anca/Fac/bio_nlp/dataset/train/train_dataset/DATA/'


train_data_list = [[] for x in range(21)]

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


    print(len(b_title), len(b_text))
    for i in range(len(b_title)):
        title = b_title[i].string
        title = ' ' if title is None else title
        text = b_text[i].string
        text = ' ' if text is None else text
        new_text = title + ' ' + text
        new_text = preprocessing(new_text)


        for no, answer in enumerate(answers):
            pair = [data[0], new_text, answer]

            train_data_list[no].append(pair)



for i, train_data in enumerate(train_data_list):
    df = pd.DataFrame(train_data, columns=['subject', 'text', 'answer'])

    df.to_csv('/home/anca/Fac/bio_nlp/dataset/new_format_dataset_cleaned/train/question_' + str(i+1) + '.csv', index=False)