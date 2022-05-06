import sys

sys.path.append('../')

from model.dataloader import DataModule

dm_arguments = {
    'train_path': '../../data/train.csv',
    'val_path': '../../data/train.csv',
    'tokenizer_name_or_path': 'bert-base-multilingual-cased',
    'train_batch_size': 4,
    'val_batch_size': 4,
}

dm = DataModule(**dm_arguments)

dm.setup()

for i, batch in enumerate(dm.train_dataloader()):
    if i==0:
        sample_input_ids = batch['input_ids']
        break

tokenizer = dm.tokenizer

decoded_text = tokenizer.batch_decode(sample_input_ids.tolist())
print(decoded_text)