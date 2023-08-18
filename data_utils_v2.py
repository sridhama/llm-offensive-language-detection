from torch.utils import data
import pandas as pd
import os, argparse, random
import torch
from torch.utils.data import Dataset, DataLoader

langs = ['english', 'danish', 'turkish', 'arabic', 'greek']
dataroot = '/content/drive/MyDrive/data/'

class OLID(Dataset):
  def __init__(self, dataroot, mode='train', lang='english'):
    if lang!='all' and lang not in langs:
      print(f'language {lang} not supported. Supported all_langs + {",".join(langs)}')
      exit() 
    if mode not in ['train', 'test']:
      print(f'mode {mode} not supported. Supported train, test')
      exit() 

    self.dataroot = dataroot 
    self.mode = mode 
    self.lang = lang 

    self.sents = []

    if lang=='all':

      for lang in langs:
        tsv_anno = os.path.join(self.dataroot, lang, f'{mode}2.tsv')
        self.df = pd.read_csv(tsv_anno, sep='\t')
        self.df = self.df[['tweet', 'subtask_a']]
        self.df = self.df.dropna()

        self.sents.extend(self.df.values.tolist())
    
    else:
      tsv_anno = os.path.join(self.dataroot, lang, f'{mode}2.tsv')
      self.df = pd.read_csv(tsv_anno, sep='\t')
      self.df = self.df[['tweet', 'subtask_a']]
      self.df = self.df.dropna()
      
      self.sents.extend(self.df.values.tolist())

    random.shuffle(self.sents)

    # total_sents = len(self.sents)
    # num_off = sum(self.df['subtask_a'] == 'OFF')
    # num_not = total_sents - num_off
    # print(f'lang: {lang}, mode:{mode} OFF:{num_off} NOT:{num_not}')
    

      # if self.mode == 'train':
      #   tsv_anno = os.path.join(self.dataroot, lang, 'train.tsv')
      #   self.df = pd.read_csv(tsv_anno, sep='\t')
      # else: # mode == test
      #   tsv_anno = os.path.join(self.dataroot, lang, 'test.tsv')
      #   self.df = pd.read_csv(tsv_anno, sep='\t')
      # self.df = self.df.dropna()
    
    # print('df size:', self.df.shape)

  def __len__(self):
    # return self.df.shape[0]
    return len(self.sents)

  def __getitem__(self, idx):
    # row = self.df.iloc[idx] 
    sent = self.sents[idx] 
    input = sent[0]
    label = sent[1]
    if label == 'OFF':
      label = 1
    else:
      label = 0 
    return {'input' : input, 'label' : label}
