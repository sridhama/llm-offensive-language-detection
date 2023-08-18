import torch
import os, sys, argparse
import numpy as np 
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from data_utils_OLID import OLID 

parser = argparse.ArgumentParser(description='BERT')
parser.add_argument("--ckpt_loc", default='english', type=str)
parser.add_argument("--language", default='english', type=str)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--subtask", default='subtask_a', type=str)

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_names = {
  'english': "bert-base-uncased",
  'danish': "Maltehb/danish-bert-botxo",
  'turkish': "dbmdz/bert-base-turkish-cased",
  'arabic': "asafaya/bert-base-arabic",
  'greek': "nlpaueb/bert-base-greek-uncased-v1"
}

model_name = model_names[args.language]

if args.subtask in ['subtask_a', 'subtask_b']:
  num_labels = 2
elif args.subtask == 'subtask_c':
  num_labels = 3
else:
  raise Exception()

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

# Download pytorch model
# model = XLMRobertaForSequenceClassification.from_pretrained(model_name,
#                                   num_labels = 2, # The number of output labels.   
#                                   output_attentions = False, # Whether the model returns attentions weights.
#                                   output_hidden_states = False, # Whether the model returns all hidden-states.
#                                   )

model.cuda()

dataroot = '/content/drive/MyDrive/data/'
print(dataroot)

lang = args.language
dataset = OLID(dataroot, 'test', lang, subtask=args.subtask)
batch_size =  dataset.df.shape[0]
testDataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
# import pdb; pdb.set_trace()
def validate_model(dataloader, subtask, tweets, label_names):
  model.eval()

  running_loss = 0.
  matches = 0.
  samples = 0.
  running_labels = []
  running_preds = []

  for idx, item in enumerate(dataloader):

    inputs = item['input']
    labels = item['label']

    input_ids = tokenizer(inputs, padding=True, truncation=True, max_length=64, return_tensors='pt')

    model_input = input_ids['input_ids'].cuda()
    attention_masks = input_ids['attention_mask'].cuda()
    labels = labels.to(device) 

    with torch.no_grad():
      outputs = model(model_input, 
                      token_type_ids=None, 
                      attention_mask=attention_masks, 
                      labels=labels)
      loss = outputs.loss

    logits = outputs.logits
    running_loss += loss.item()

    preds = torch.argmax(logits, dim=1)

    running_labels.extend( list(labels.detach().cpu().numpy()) )
    running_preds.extend( list(preds.detach().cpu().numpy()) )

    running_loss += loss.item()

    matches += torch.sum(preds == labels).item()
    samples += labels.shape[0]

    torch.cuda.empty_cache()

    # break

  # print(running_labels)
  if subtask == 'subtask_c':
    average = 'macro'
  elif subtask in ['subtask_a', 'subtask_b']:
    average = 'binary'
  else:
    raise Exception()

  precision = precision_score(running_labels, running_preds, average=average)
  recall = recall_score(running_labels, running_preds, average=average)
  f1 = f1_score(running_labels, running_preds, average=average)
  running_loss /= len(dataloader)
  running_acc = matches / samples

  if subtask == 'subtask_a':
    label_dict = {0: 'NOT', 1: 'OFF'}
  elif subtask == 'subtask_b':
    label_dict = {0: 'UNT', 1: 'TIN'}
  elif subtask == 'subtask_c':
    label_dict = {0: 'IND', 1: 'GRP', 2: 'OTH'}
  else:
        raise Exception()
      

  for t, p, tweet, lbl in zip(running_labels, running_preds, tweets, label_names):
    if t != p:
      print(f'true label: {label_dict[t]}\t pred_label: {label_dict[p]}\t tweet: {tweet}')

  print(confusion_matrix(running_labels, running_preds))

  return precision, recall, f1, running_loss, running_acc


model.load_state_dict(torch.load(args.ckpt_loc)['best_model_wts'])
model.eval()
precision, recall, f1, val_loss, val_acc = validate_model(testDataloader, subtask=args.subtask, tweets=dataset.df['tweet'], label_names=dataset.df[args.subtask])
print(f'{precision}\t{recall}\t{f1}\t{val_acc}')