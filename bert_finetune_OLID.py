import torch
import os, sys, argparse
import numpy as np 
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

from sklearn.metrics import f1_score, precision_score, recall_score

from data_utils_OLID import OLID 

parser = argparse.ArgumentParser(description='BERT')
parser.add_argument("--logs_dir", default='english', type=str)
parser.add_argument("--language", default='english', type=str)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--learning_rate", default=2e-5, type=float)
parser.add_argument("--weight_decay", default=1e-2, type=float)
parser.add_argument("--epochs", default=50, type=int)
parser.add_argument("--subtask", default='subtask_a', type=str)

args = parser.parse_args()
args.logs_dir += f'/{args.language}_{args.subtask}'
if not os.path.exists(args.logs_dir):
  os.mkdir(args.logs_dir)
  print(f'{args.logs_dir} created!')

logs_file = os.path.join(args.logs_dir, 'logs.txt')

def logprint(log):
    print(log, end='')
    with open(logs_file, 'a') as f:
        f.write(log) 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the model repo
# model_name = "xlm-roberta-base" 
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

def warmp_up(lr, epoch, warmup_epochs):
  lr = lr * (epoch / warmup_epochs)
  return lr

def step_down(lr, epoch, step_down_epochs):
  lr = (1 - epoch / step_down_epochs) * lr 
  return lr 

dataroot = '/content/drive/MyDrive/data/'

lang = args.language
batch_size = args.batch_size
dataset = OLID(dataroot, 'train', lang, subtask=args.subtask)
trainDataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

dataset = OLID(dataroot, 'test', lang, subtask=args.subtask)
testDataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

for idx, item in enumerate(trainDataloader):
  inputs = item['input']
  labels = item['label']
  print('inputs:', inputs)
  print('labels:', labels)
  break

weight_decay = args.weight_decay
learning_rate = args.learning_rate 

optimizer = torch.optim.AdamW(model.parameters(),
                  lr = learning_rate, 
                  eps = 1e-8,
                  weight_decay=weight_decay,
                )

epochs = args.epochs

lr_warmup_epochs = 0.1 * epochs 
init_lr = 0 

def validate_model(dataloader, subtask):
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

  # import pdb; pdb.set_trace()
  precision = precision_score(running_labels, running_preds, average=average)
  recall = recall_score(running_labels, running_preds, average=average)
  f1 = f1_score(running_labels, running_preds, average=average)
  running_loss /= len(dataloader)
  running_acc = matches / samples 

  return precision, recall, f1, running_loss, running_acc

def train_model(dataloader):

  model.train()

  running_loss = 0.
  matches = 0.
  samples = 0.

  for idx, item in enumerate(dataloader):

    inputs = item['input']
    labels = item['label']

    input_ids = tokenizer(inputs, padding=True, truncation=True, max_length=64, return_tensors='pt')

    model_input = input_ids['input_ids'].to(device)
    attention_masks = input_ids['attention_mask'].to(device)
    labels = labels.to(device) 

    model.zero_grad()

    outputs = model(model_input, 
                    token_type_ids=None, 
                    attention_mask=attention_masks, 
                    labels=labels)
    loss = outputs.loss

    loss.backward()
    optimizer.step()

    logits = outputs.logits
    running_loss += loss.item()

    preds = torch.argmax(logits, dim=1)

    matches += torch.sum(preds == labels).item()
    samples += preds.shape[0]

    torch.cuda.empty_cache()

    # break

  running_loss /= len(dataloader)
  running_acc = matches / samples

  return running_loss, running_acc  


best_val_f1 = 0. 

for epoch_i in range(epochs):

  logprint("\n")
  logprint('======== Epoch {:} / {:} ========\n'.format(epoch_i + 1, epochs))
  logprint('Training...\n')

  if epoch_i < lr_warmup_epochs:
    lr = warmp_up(learning_rate, epoch_i, lr_warmup_epochs)
  else:
    lr = step_down(learning_rate, epoch_i - lr_warmup_epochs, epochs - lr_warmup_epochs)
  for g in optimizer.param_groups:
    g['lr'] = lr
  
  train_loss, train_acc = train_model(trainDataloader)

  logprint(f"Total loss: {train_loss} Train Acc: {train_acc}\n")

  precision, recall, f1, val_loss, val_acc = validate_model(testDataloader, subtask=args.subtask)

  logprint(f"Validation Loss: {val_loss}: Acc: {val_acc}\n")
  logprint(f'Validation Precision:{precision} Recall: {recall} F1: {f1}\n')

  if f1 > best_val_f1:
    best_val_f1 = f1 

    logprint(f'best epoch: {epoch_i+1} best f1: {f1}\n')
    logprint(f'{precision}\t{recall}\t{f1}\t{val_acc}')

    save_dict = {
      'best_model_wts' : model.state_dict(),
      'best_f1' : best_val_f1,
      'precision': precision,
      'recall': recall
    }

    save_model_path = os.path.join(args.logs_dir, f'bert_{args.language}.pth')

    torch.save(save_dict, save_model_path)