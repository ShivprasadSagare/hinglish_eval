from turtle import shape
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer
import pandas as pd
import torch

class DS(Dataset):
    def __init__(self, data_path, tokenizer):
        self.df = pd.read_csv(data_path)
        self.df = self.df[:len(self.df)]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        target_text = self.df.iloc[idx]['Hinglish']

        target_encoding = self.tokenizer(target_text, return_tensors='pt', padding='max_length', truncation=True)
        input_ids, attention_mask = target_encoding['input_ids'], target_encoding['attention_mask']

        # Average rating Disagreement
        if 'Average rating' in self.df.columns:
            # label = self.df.iloc[idx]['Average rating']
            label = self.df.iloc[idx]['Disagreement']
            label_encoding = torch.zeros(10)
            label_encoding[label-1] = 1
        else:
            label_encoding = torch.zeros(10)

        return {'input_ids': input_ids.squeeze(), 'attention_mask': attention_mask.squeeze(), 'labels': label_encoding}    


class DataModule(pl.LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer_name_or_path)

    def setup(self, stage=None):
        if stage == 'fit':
            self.train = DS(self.hparams.train_path, self.tokenizer)
            self.val = DS(self.hparams.val_path, self.tokenizer)
        else:
            self.test = DS(self.hparams.test_path, self.tokenizer)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.hparams.train_batch_size, num_workers=0,shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.hparams.val_batch_size, num_workers=0,shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.hparams.test_batch_size, num_workers=0,shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.test, batch_size=self.hparams.test_batch_size, num_workers=8,shuffle=False)

    @staticmethod
    def add_datamodule_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('Datamodule args')
        parser.add_argument('--train_path', default='data/train.csv', type=str)
        parser.add_argument('--val_path', default='data/val.csv', type=str)
        # parser.add_argument('--test_path', default='data/test.csv', type=str)
        parser.add_argument('--tokenizer_name_or_path', type=str)
        # parser.add_argument('--max_source_length', type=int, default=128)
        # parser.add_argument('--max_target_length', type=int, default=128)
        parser.add_argument('--train_batch_size', type=int, default=4)
        parser.add_argument('--val_batch_size', type=int, default=4)
        # parser.add_argument('--test_batch_size', type=int, default=4)
        return parent_parser