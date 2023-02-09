import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm 
from torch.utils.data import Dataset, DataLoader


class Dataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_length=512):
        self.tokenizer = tokenizer                      # the Tokenizer model
        self.data      = dataframe                      # the full dataset
        self.report    = dataframe.report               # the text data (i.e., the reports)
        self.max_len   = max_length                     # max length fro truncation

    def __len__(self):
        return len(self.report)

    def __getitem__(self, index):
        report = str(self.report[index])
        report = " ".join(report.split())

        inputs = self.tokenizer.encode_plus(
            report,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
        }

class Train_Dataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_length=512):
        self.tokenizer = tokenizer                      # the Tokenizer model
        self.data      = dataframe                      # the full dataset
        self.report    = dataframe.report               # the text data (i.e., the reports)
        self.targets   = self.data.Report_as_Multilabel # labels' list to classify
        self.max_len   = max_length                     # max length fro truncation

    def __len__(self):
        return len(self.report)

    def __getitem__(self, index):
        report = str(self.report[index])
        report = " ".join(report.split())

        inputs = self.tokenizer.encode_plus(
            report,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }
    