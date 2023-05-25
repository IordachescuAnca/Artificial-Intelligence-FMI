# import libraries
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW
import time
from sklearn.preprocessing import LabelEncoder
import datetime
import os



def transform_label(label):
  if label == '0':
     return 0
  if label == '1a':
     return 1
  if label == '1b':
     return 2
  if label == '2a':
     return 3
  if label == '2b':
     return 4
  if label == '3a':
     return 5
  if label == '3b':
     return 6

# USEFUL FUNCTIONS
def accuracy(y_pred, y_test):
  acc = (torch.log_softmax(y_pred, dim=1).argmax(dim=1) == y_test).sum().float() / float(y_test.size(0))
  return acc


def format_time(elapsed):
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def processdata(tweets,labels, tokenizer):
  input_ids = []
  attention_masks = []
  for tweet in tweets:
    encoded_dict = tokenizer.encode_plus(
                        tweet,                     
                        add_special_tokens = True,
                        max_length = 256,           
                        pad_to_max_length = True,
                        return_attention_mask = True,   
                        return_tensors = 'pt',   
                   )
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

  input_ids = torch.cat(input_ids, dim=0)
  attention_masks = torch.cat(attention_masks, dim=0)
  labels = torch.tensor(labels)
  return input_ids,attention_masks,labels







def train(model, train_loader, val_loader, optimizer,scheduler, EPOCHS, device, checkpoint_path):  
  best_acc = -1
  total_step = len(train_loader)

  for epoch in range(EPOCHS):
    # Measure how long the training epoch takes.
    train_start = time.time()
    model.train()

    # Reset the total loss and accuracy for this epoch.
    total_train_loss = 0
    total_train_acc  = 0
    for batch_idx, (pair_token_ids, mask_ids, y) in enumerate(train_loader):

      # Unpack this training batch from our dataloader. 
      pair_token_ids = pair_token_ids.to(device)
      mask_ids = mask_ids.to(device)
      labels = y.to(device)

      #clear any previously calculated gradients before performing a backward pass
      optimizer.zero_grad()

      #Get the loss and prediction
      loss, prediction = model(pair_token_ids, 
                             attention_mask=mask_ids, 
                             labels=labels).values()
    

      acc = accuracy(prediction, labels)
      
      # Accumulate the training loss and accuracy over all of the batches so that we can
      # calculate the average loss at the end
      total_train_loss += loss.item()
      total_train_acc  += acc.item()

      # Perform a backward pass to calculate the gradients.
      loss.backward()

      # Clip the norm of the gradients to 1.0.
      # This is to help prevent the "exploding gradients" problem.
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

      # Update parameters and take a step using the computed gradient.
      optimizer.step()

      # Update the learning rate.
      scheduler.step()

    # Calculate the average accuracy and loss over all of the batches.
    train_acc  = total_train_acc/len(train_loader)
    train_loss = total_train_loss/len(train_loader)
    train_end = time.time()

    # Put the model in evaluation mode
    model.eval()

    total_val_acc  = 0
    total_val_loss = 0
    val_start = time.time()
    with torch.no_grad():
      for batch_idx, (pair_token_ids, mask_ids, y) in enumerate(val_loader):

        #clear any previously calculated gradients before performing a backward pass
        optimizer.zero_grad()

        # Unpack this validation batch from our dataloader. 
        pair_token_ids = pair_token_ids.to(device)
        mask_ids = mask_ids.to(device)
        labels = y.to(device)
        
        #Get the loss and prediction
        loss, prediction = model(pair_token_ids, 
                             attention_mask=mask_ids, 
                             labels=labels).values()

        # Calculate the accuracy for this batch
        acc = accuracy(prediction, labels)
        # Accumulate the validation loss and Accuracy
        total_val_loss += loss.item()
        total_val_acc  += acc.item()

    # Calculate the average accuracy and loss over all of the batches.
    val_acc  = total_val_acc/len(val_loader)
    val_loss = total_val_loss/len(val_loader)

    #end = time.time()
    val_end = time.time()
    hours, rem = divmod(val_end-train_start, 3600)
    minutes, seconds = divmod(rem, 60)

    print(f'Epoch {epoch+1}: train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}')
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


    if best_acc <= val_acc:
        best_acc = val_acc
        model.save_pretrained(checkpoint_path + '/' + str(epoch) + '.pth')





checkpoints_output = '/mnt/storage/Fac/ckpts_roberta/'
os.makedirs(checkpoints_output, exist_ok = True)

root_train = '/home/anca/Fac/bio_nlp/dataset/new_format_dataset/train/'


for i in range(1, 22):
   
    file = root_train + 'question_' + str(i) + '.csv'
    print('*****')
    print(file)

    ckpt_output_train = checkpoints_output + 'question_' + str(i)
    os.makedirs(ckpt_output_train, exist_ok = True)

    model_name = 'distilroberta-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)


    # Split dataset - 80% train 20% validation

    train_df = pd.read_csv(file)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_df['text'].values.tolist(), train_df['answer'].values.tolist(),
        stratify = train_df['answer'], test_size=0.2
    )

    if i == 16 or i == 18:
       val_labels = [transform_label(x) for x in val_labels]
       train_labels = [transform_label(x) for x in train_labels]
    else:
      val_labels = [int(x) for x in val_labels]
      train_labels = [int(x) for x in train_labels]


    #Preprocess dataset using tokenizer
    train_input_ids,train_attention_masks,train_labels = processdata(train_texts,train_labels, tokenizer)
    val_input_ids,val_attention_masks,val_labels = processdata(val_texts,val_labels, tokenizer)

    print(len(val_labels), len(train_labels))
    # Create dataset & dataloader
    batch_size = 32
    train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
    val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_labels)

    train_dataloader = DataLoader(
                train_dataset, 
                sampler = RandomSampler(train_dataset),
                batch_size = batch_size
            )

    validation_dataloader = DataLoader(
                val_dataset, 
                sampler = SequentialSampler(val_dataset), 
                batch_size = batch_size
            )


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(np.unique(val_labels)), ignore_mismatched_sizes=True)
    model.to(device)


    optimizer = AdamW(model.parameters(),
                    lr = 3e-5, 
                    eps = 1e-8)

    EPOCHS = 3
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, 
                                                num_training_steps = total_steps)
    

    train(model, train_dataloader, validation_dataloader, optimizer, scheduler, EPOCHS=EPOCHS, device=device, checkpoint_path=ckpt_output_train)

    

