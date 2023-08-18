from torch.utils import data
import pandas as pd
import os 
import torch
from torch.utils.data import Dataset, DataLoader

langs = ['english', 'danish', 'turkish', 'arabic', 'greek']
dataroot = '/content/drive/MyDrive/data/'

class OLID(Dataset):
  def __init__(self, dataroot, mode='train', lang='english'):
    if lang not in langs:
      print(f'language {lang} not supported. Supported {",".join(langs)}')
      exit() 

    self.dataroot = dataroot 
    self.mode = mode 
    self.lang = lang 

    self.sents = {}

    if self.mode == 'train':
      tsv_anno = os.path.join(self.dataroot, lang, 'train.tsv')
      self.df = pd.read_csv(tsv_anno, sep='\t')
    else: # mode == test
      tsv_anno = os.path.join(self.dataroot, lang, 'test.tsv')
      self.df = pd.read_csv(tsv_anno, sep='\t')

  def __len__(self):
    return self.df.shape[0]

  def __getitem__(self, idx):
    row = self.df.iloc[idx] 
    input = row['tweet']
    label = row['subtask_a']
    if label == 'OFF':
      label = 1
    else:
      label = 0 
    return {'input' : input, 'label' : label}

# dataset = OLID(dataroot, 'train', 'english')
# trainDataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# dataset = OLID(dataroot, 'test', 'english')
# testDataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# for idx, item in enumerate(trainDataloader):
#   inputs = item['input']
#   labels = item['label']
#   print('inputs:',inputs)
#   print('labels:',labels)
#   break