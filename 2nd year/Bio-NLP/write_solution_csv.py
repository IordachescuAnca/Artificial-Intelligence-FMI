import random
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, BertForSequenceClassification

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

model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
arabert_model = AutoModel.from_pretrained(model_name)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = AutoModelForSequenceClassification.from_pretrained('/home/anca/Fac/nlp2/ckpt/bert/5.pth')
model.to(device)
model.eval()

dev = pd.read_csv('data/test_data.csv')

dev_texts = dev.text.values
dev_labels = dev.rewire_id.values
input_ids,attention_masks = processdata(dev_texts)
batch_size = 16 

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
print(len(pred_flat))

file_name = '/home/anca/Fac/nlp2/SUBMISSION_{}_test.csv'.format('bert')
file = open(file_name, 'w')
file.write('user_id,label_pred\n')
for i in range(len(dev_labels)):
    id = dev_labels[i]
    pred = pred_flat[i]
    line = id + ',' + pred + '\n'
    file.write(line)

file.close()
