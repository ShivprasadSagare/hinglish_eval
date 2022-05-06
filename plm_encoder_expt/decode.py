from model.model import FineTuner 
from model.dataloader import DataModule

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import argparse
import os
from datetime import datetime
# hinglish/plm_encoder_expt/experiments/2dl6a5rb/checkpoints/epoch=13-step=3877.ckpt
ckpt_path = './experiments/2dl6a5rb/checkpoints/epoch=13-step=3877.ckpt'
model = FineTuner.load_from_checkpoint(ckpt_path)
trainer = pl.Trainer(gpus=1, strategy='ddp')

dm = DataModule(
    train_path='../data/train_new.csv',
    val_path='../data/val_new.csv',
    test_path='../data/valid.csv',
    tokenizer_name_or_path='bert-base-multilingual-cased',
    train_batch_size=4,
    val_batch_size=4,
    test_batch_size=4,
    stage = 'test'
)
predictions = trainer.predict(model, datamodule=dm)
results = []
for prediction in predictions:
    results.extend(prediction['predictions'])

fp = 'results_mbert_3e4_30epochs.txt'
with open(fp, 'a') as f:
    for result in results:
        f.write(str(result) + '\n')