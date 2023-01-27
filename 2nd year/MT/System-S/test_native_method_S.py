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

def processdata(tweets):
  input_ids = []
  attention_masks = []
  for tweet in tweets:
    encoded_dict = tokenizer.encode_plus(
                        tweet,                     
                        add_special_tokens = True,
                        max_length = 128,           
                        pad_to_max_length = True,
                        return_attention_mask = True,   
                        return_tensors = 'pt',   
                   )
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

  input_ids = torch.cat(input_ids, dim=0)
  attention_masks = torch.cat(attention_masks, dim=0)

  return input_ids,attention_masks


model_name = 'dccuchile/bert-base-spanish-wwm-cased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
arabert_model = AutoModel.from_pretrained(model_name)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = AutoModelForSequenceClassification.from_pretrained('/home/anca/Fac/MT/System-S/checkpoints/8.pth')
model.to(device)
model.eval()


dev = pd.read_csv('/home/anca/Fac/MT/test dataset/spanish.csv')

dev_texts = dev.tweet.values
dev_labels = dev.label.values
input_ids,attention_masks = processdata(dev_texts)
batch_size = 8 

prediction_data = TensorDataset(input_ids, attention_masks)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size, shuffle = False)

outputs = []
for batch_idx, (pair_token_ids, mask_ids) in enumerate(prediction_dataloader):
    pair_token_ids = pair_token_ids.to(device)
    mask_ids = mask_ids.to(device)
    output = model(pair_token_ids, attention_mask=mask_ids)['logits']
    output = output.cpu().detach().numpy()

    outputs.append(output)

outputs = [x for y in outputs for x in y]
pred_flat = np.argmax(outputs, axis=1).flatten()

print(accuracy_score(dev_labels, pred_flat))
print(recall_score(dev_labels, pred_flat))
print(precision_score(dev_labels, pred_flat))