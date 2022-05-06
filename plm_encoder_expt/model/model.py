import pytorch_lightning as pl
from sklearn.metrics import f1_score
from transformers import AutoModel
import torch
import numpy as np


class FineTuner(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModel.from_pretrained(
            self.hparams.model_name_or_path
        )
        self.layer_1 = torch.nn.Linear(self.model.config.hidden_size, 10)
        self.criterion = torch.nn.BCELoss()

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        outputs = self.layer_1(outputs.pooler_output)
        outputs = torch.sigmoid(outputs)
        loss = 0
        if labels is not None:
            loss = self.criterion(outputs, labels)
        return loss, outputs

    def _step(self, batch):
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
        loss, outputs = self(input_ids, attention_mask, labels)
        return loss, outputs

    def training_step(self, batch, batch_idx):
        loss, outputs = self._step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, outputs = self._step(batch)
        self.log("val_loss", loss, on_step=True, on_epoch=True)

        return {'val_loss': loss, 'outputs': outputs, 'labels': batch['labels']}

    def validation_epoch_end(self, outputs):
        # calculate metric score here
        # predictions = [x['outputs'] for x in outputs]
        # predictions = np.array([np.argmax(preds.cpu().numpy(), axis=1) for preds in predictions]).flatten()
        # labels = [x['labels'] for x in outputs]
        # labels = np.array([np.argmax(labels.cpu().numpy(), axis=1) for labels in labels]).flatten()
        # print(predictions)
        # print(labels)
        # f1 = f1_score(labels, predictions, average='weighted')
        # self.log("val_f1", f1)
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def predict_step(self, batch, batch_idx: int):
        loss, outputs = self._step(batch)
        predictions = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        return {'predictions': predictions}


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('Model args')
        parser.add_argument('--learning_rate', default=2e-5, type=float)
        parser.add_argument('--model_name_or_path', default='t5-base', type=str)
        return parent_parser