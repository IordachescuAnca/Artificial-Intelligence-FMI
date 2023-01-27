import random
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, BertForSequenceClassification
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import coverage_error

def processdata(tweets):
  input_ids = []
  attention_masks = []
  for tweet in tweets:
    encoded_dict = tokenizer.encode_plus(
                        tweet,                    
                        add_special_tokens = True,
                        max_length = 150,           
                        pad_to_max_length = True,
                        return_attention_mask = True,   
                        return_tensors = 'pt',     
                   )
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

  input_ids = torch.cat(input_ids, dim=0)
  attention_masks = torch.cat(attention_masks, dim=0)
  return input_ids,attention_masks


model_name = 'mental/mental-bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = AutoModelForSequenceClassification.from_pretrained('/home/anca/Fac/MT/System 3/checkpoints/0.pth')
model.to(device)
model.eval()

dev = pd.read_csv('/home/anca/Fac/MT/Translated - Helsinki/spanish.csv')

dev_texts = dev.tweet.values
dev_labels = dev.label.values
input_ids,attention_masks = processdata(dev_texts)
batch_size = 8 

predicted = []
with torch.no_grad():
    for i, text in enumerate(dev_texts):
        inputs = tokenizer(text, add_special_tokens = True,
                        max_length = 150,           
                        pad_to_max_length = True,
                        return_attention_mask = True,   
                        return_tensors = 'pt')
        inputs = inputs.to(device)
        logits = model(**inputs).logits
        predicted_class_id = logits.argmax().item()

        predicted.append(predicted_class_id)

print(accuracy_score(dev_labels, predicted))
print(recall_score(dev_labels, predicted))
print(precision_score(dev_labels, predicted))

print(predicted)
print(dev_labels)