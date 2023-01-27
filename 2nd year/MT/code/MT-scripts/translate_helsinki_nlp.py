# -*- coding: utf-8 -*-



from google.colab import drive
drive.mount('/content/drive/')

#!pip install --no-cache-dir transformers sentencepiece

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

df_test = pd.read_csv('/content/drive/MyDrive/Machine Translation/Spanish/train.csv')

df_test

#model_name = "Helsinki-NLP/opus-mt-jap-en"
#model_name = "Helsinki-NLP/opus-mt-ko-en"
#model_name = "Helsinki-NLP/opus-mt-ru-en"
model_name = "Helsinki-NLP/opus-mt-es-en"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
     


model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

lines = []

for index, row in df_test.iterrows():
  text = row['tweet']
  label = row['label']

  try:
    tok_text = tokenizer([text], return_tensors="pt", max_length = 512)
    text_ids = model.generate(**tok_text)
    translate = tokenizer.batch_decode(text_ids, skip_special_tokens=True)[0]
    lines.append([translate, text, label])

  except:
    print('exp')

df = pd.DataFrame(lines, columns=['translate', 'tweet', 'label'])
df.to_csv('/content/drive/MyDrive/Machine Translation/Translated - English/spanish.csv')