# _*_ coding: utf-8 _*_

import torch
import torch.nn as nn
from torch.nn import functional as F

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

import pickle
import pandas as pd
from sklearn.metrics import classification_report, f1_score

doc_list = ['frightening', 'alcohol','nudity', 'violence', 'profanity']
working_aspect = doc_list[0]
print('Now working on:', working_aspect)
base_dir = '/home/yzhan273/Research/MPAA/Severity_Class_Pred/sent_bert/data_sent_emb/'
train_file = base_dir + working_aspect + '_train.pkl'
dev_file = base_dir + working_aspect + '_dev.pkl'

# _*_ coding: utf-8 _*_

class LSTM_model(pl.LightningModule):
	def __init__(self, input_size, output_size, hidden_size):
		super(LSTM_model, self).__init__()

		bsz = 1
		self.direction = 2
		self.input_size = input_size
		self.output_size = output_size
		self.hidden_size = hidden_size
		self.batch_size = bsz
		self.lstm = nn.LSTM(self.input_size, self.hidden_size, num_layers=1, batch_first=True, bidirectional=True)
		# self.label = nn.Linear(self.hidden_size, self.output_size) # single direction
		self.label = nn.Linear(self.hidden_size * self.direction, self.output_size)

	def forward(self, input_sentence, batch_size=None):
		# print("here")
		if batch_size is None:
			# Initial hidden state of the LSTM (num_layers * num_directions, batch, hidden_size)
			h_0 = torch.zeros(1 * self.direction, self.batch_size, self.hidden_size).requires_grad_().to(
				device='cuda:6')
			# Initial cell state of the LSTM
			c_0 = torch.zeros(1 * self.direction, self.batch_size, self.hidden_size).requires_grad_().to(
				device='cuda:6')

		else:
			h_0 = torch.zeros(1 * self.direction, batch_size, self.hidden_size).requires_grad_().to(device='cuda:6')
			c_0 = torch.zeros(1 * self.direction, batch_size, self.hidden_size).requires_grad_().to(device='cuda:6')

		output, (final_hidden_state, final_cell_state) = self.lstm(input_sentence, (h_0, c_0))

		output = pad_packed_sequence(output, batch_first=True)  # padded seq, lengths
		output = torch.max(output[0], dim=1)[0]  # after max, (max tensor, max_indices)

		final_output = self.label(output)

		return final_output

	def loss_function(self, prediction, target):
		return F.cross_entropy(prediction, target)

	def training_step(self, batch, batch_idx):
		text, text_len, target = batch

		prediction = self(text, len(text_len))
		loss = self.loss_function(prediction, target)

		return {'loss': loss}

	def validation_step(self, batch, batch_idx):
		text, text_len, target = batch

		prediction = self(text, len(text_len))
		val_loss = self.loss_function(prediction, target)
		prediction_digits = [torch.argmax(x).item() for x in prediction]

		return {'prediction_digits': prediction_digits, 'target': target.tolist(), 'val_loss': val_loss}

	def validation_epoch_end(self, val_step_outputs):
		avg_val_loss = torch.tensor([x['val_loss'] for x in val_step_outputs]).mean()
		val_predictions = [x['prediction_digits'] for x in val_step_outputs]
		val_targets = [x['target'] for x in val_step_outputs]
		# unpack list of lists
		val_predictions = [item for sublist in val_predictions for item in sublist]
		val_targets = [item for sublist in val_targets for item in sublist]
		# print(val_predictions)
		# print(val_targets)
		print(classification_report(val_targets, val_predictions, digits=4))
		val_weighted_f1 = f1_score(val_targets, val_predictions, average='weighted')

		return {'avg_val_loss': avg_val_loss, 'val_weighted_f1': val_weighted_f1}

	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters(), lr=1e-3)

	def train_dataloader(self):
		train_raw_data = pd.read_pickle(train_file)
		train_dataset = MovieScriptDataset(train_raw_data)
		train_loader = torch.utils.data.DataLoader(
			train_dataset, batch_size=20, shuffle=True, collate_fn=my_collate_fn, num_workers=10)
		return train_loader

	def val_dataloader(self):
		dev_raw_data = pd.read_pickle(dev_file)
		dev_dataset = MovieScriptDataset(dev_raw_data)
		dev_loader = torch.utils.data.DataLoader(
			dev_dataset, batch_size=1, shuffle=False, collate_fn=my_collate_fn)
		return dev_loader


class MovieScriptDataset(torch.utils.data.Dataset):
	def __init__(self, tabular):
		if isinstance(tabular, str):
			self.annotations = pd.read_csv(tabular, sep='\t')
		else:
			self.annotations = tabular

	def __len__(self):
		return len(self.annotations)

	def __getitem__(self, index):
		text = self.annotations.iloc[index, -1]  # -1 is sent emb index
		y_label = torch.tensor(int(self.annotations.iloc[index, -2]))  # -2 is label index
		return {
			'text': text,
			'label': y_label
		}


def my_collate_fn(batch):
	text_batch = [each_item['text'] for each_item in batch]
	label_batch = [each_item['label'] for each_item in batch]
	data_length = [len(sq) for sq in text_batch]
	# sort from longest to shortest
	text_batch = [x for _, x in sorted(zip(data_length, text_batch), key=lambda pair: pair[0], reverse=True)]
	label_batch = torch.stack(
		[x for _, x in sorted(zip(data_length, label_batch), key=lambda pair: pair[0], reverse=True)])
	data_length.sort(reverse=True)
	# pad
	text_batch = pad_sequence(text_batch, batch_first=True, padding_value=0)
	# pack padded
	text_batch = pack_padded_sequence(text_batch, data_length, batch_first=True)
	return text_batch, data_length, label_batch


if __name__ == "__main__":
	output_size = 4
	input_size = 768
	hidden_size = 200
	training_epochs = 30

	model = LSTM_model(input_size, output_size, hidden_size)

	early_stop_callback = EarlyStopping(
		monitor='val_weighted_f1',
		min_delta=0.00,
		patience=10,
		verbose=False,
		mode='max'
	)
    # 3. Init ModelCheckpoint callback, monitoring 'val_loss'
	checkpoint_callback = ModelCheckpoint(
        monitor='val_weighted_f1',
        filepath='/home/yzhan273/Research/MPAA/Severity_Class_Pred/sent_bert/LT_save_model/',
        mode='max')


	trainer = pl.Trainer(fast_dev_run=False, max_epochs=training_epochs, gpus=[6],
						 early_stop_callback=early_stop_callback,
                        checkpoint_callback=checkpoint_callback)
	trainer.fit(model)

# output, final_hidden_state, final_cell_state = model(torch.rand(5, 10, 768), batch_size=5)
# print(output.shape, final_hidden_state.shape, final_cell_state.shape)

# best          precision    recall  f1-score   support
#
#            0     0.5962    0.4697    0.5254        66
#            1     0.5179    0.5321    0.5249       109
#            2     0.5435    0.5952    0.5682       126
#            3     0.4516    0.4444    0.4480        63
#
#     accuracy                         0.5275       364
#    macro avg     0.5273    0.5104    0.5166       364
# weighted avg     0.5295    0.5275    0.5267       364