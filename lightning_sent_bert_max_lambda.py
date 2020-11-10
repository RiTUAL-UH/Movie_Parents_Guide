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
print("here!")
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
    def __init__(self, input_size, slate_num, hidden_size):
        super(LSTM_model, self).__init__()

        bsz = 1
        self.direction = 2
        self.input_size = input_size
        self.slate_num = slate_num
        self.hidden_size = hidden_size
        self.batch_size = bsz
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        # self.label = nn.Linear(self.hidden_size, self.output_size) # single direction

        self.ranker = nn.Linear(self.hidden_size * self.direction * self.slate_num, self.slate_num)
        
    def forward_one(self, x, batch_size = None):
        # print("here")
        # x dim (seq_len * embedding dim)
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
        # x dim add one dummy batch size (1 * seq_len * embedding dim)
        output, (final_hidden_state, final_cell_state) = self.lstm(x.unsqueeze(0), (h_0, c_0))

        output = torch.max(output[0], dim=0)[0]  # after max, (max tensor, max_indices)
        
        return output

    def forward(self, x, batch_size=None):
        list_for_rank = []
        for i in range(len(x)):
            list_for_rank.append(self.forward_one(x[i]))
            
#         out1 = self.forward_one(x[0])
#         out2 = self.forward_one(x[1])
#         out3 = self.forward_one(x[2])
#         out4 = self.forward_one(x[3])
#         out5 = self.forward_one(x[4])
        championship = torch.cat(list_for_rank)
        final_out = self.ranker(championship)

        return final_out

    def loss_function(self, y_pred, y_true, eps=1e-10, padded_value_indicator=-1, weighing_scheme=None, k=None, sigma=1., mu=10.,
                   reduction="sum", reduction_log="binary"):
        """
        LambdaLoss framework for LTR losses implementations, introduced in "The LambdaLoss Framework for Ranking Metric Optimization".
        Contains implementations of different weighing schemes corresponding to e.g. LambdaRank or RankNet.
        :param y_pred: predictions from the model, shape [batch_size, slate_length]
        :param y_true: ground truth labels, shape [batch_size, slate_length]
        :param eps: epsilon value, used for numerical stability
        :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
        :param weighing_scheme: a string corresponding to a name of one of the weighing schemes
        :param k: rank at which the loss is truncated
        :param sigma: score difference weight used in the sigmoid function
        :param mu: optional weight used in NDCGLoss2++ weighing scheme
        :param reduction: losses reduction method, could be either a sum or a mean
        :param reduction_log: logarithm variant used prior to masking and loss reduction, either binary or natural
        :return: loss value, a torch.Tensor
        """
        device = y_pred.device
        y_pred = y_pred.clone()
        y_true = y_true.clone()

        padded_mask = y_true == padded_value_indicator
        y_pred[padded_mask] = float("-inf")
        y_true[padded_mask] = float("-inf")

        # Here we sort the true and predicted relevancy scores.
        y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
        y_true_sorted, _ = y_true.sort(descending=True, dim=-1)

        # After sorting, we can mask out the pairs of indices (i, j) containing index of a padded element.
        true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
        true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
        padded_pairs_mask = torch.isfinite(true_diffs)

        if weighing_scheme != "ndcgLoss1_scheme":
            padded_pairs_mask = padded_pairs_mask & (true_diffs > 0)

        ndcg_at_k_mask = torch.zeros((y_pred.shape[1], y_pred.shape[1]), dtype=torch.bool, device=device)
        ndcg_at_k_mask[:k, :k] = 1

        # Here we clamp the -infs to get correct gains and ideal DCGs (maxDCGs)
        true_sorted_by_preds.clamp_(min=0.)
        y_true_sorted.clamp_(min=0.)

        # Here we find the gains, discounts and ideal DCGs per slate.
        pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device)
        D = torch.log2(1. + pos_idxs.float())[None, :]
        maxDCGs = torch.sum(((torch.pow(2, y_true_sorted) - 1) / D)[:, :k], dim=-1).clamp(min=eps)
        G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

        # Here we apply appropriate weighing scheme - ndcgLoss1, ndcgLoss2, ndcgLoss2++ or no weights (=1.0)
        if weighing_scheme is None:
            weights = 1.
        else:
            weights = globals()[weighing_scheme](G, D, mu, true_sorted_by_preds)  # type: ignore

        # We are clamping the array entries to maintain correct backprop (log(0) and division by 0)
        scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :]).clamp(min=-1e8, max=1e8)
        scores_diffs[torch.isnan(scores_diffs)] = 0.
        weighted_probas = (torch.sigmoid(sigma * scores_diffs).clamp(min=eps) ** weights).clamp(min=eps)
        if reduction_log == "natural":
            losses = torch.log(weighted_probas)
        elif reduction_log == "binary":
            losses = torch.log2(weighted_probas)
        else:
            raise ValueError("Reduction logarithm base can be either natural or binary")

        masked_losses = losses[padded_pairs_mask & ndcg_at_k_mask]
        if reduction == "sum":
            loss = -torch.sum(masked_losses)
        elif reduction == "mean":
            loss = -torch.mean(masked_losses)
        else:
            raise ValueError("Reduction method can be either sum or mean")

        return loss

    def training_step(self, batch, batch_idx):
        text, target = batch

        prediction = self(text)
        # dummy batch
        loss = self.loss_function(prediction.unsqueeze(0), target.unsqueeze(0))

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        text, target = batch

        prediction = self(text)
        # dummy batch
        val_loss = self.loss_function(prediction.unsqueeze(0), target.unsqueeze(0))

        return {'val_loss': val_loss}

    def validation_epoch_end(self, val_step_outputs):
        avg_val_loss = torch.tensor([x['val_loss'] for x in val_step_outputs]).mean()
        print("Validation loss:", avg_val_loss.item())

        return {'avg_val_loss': avg_val_loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def train_dataloader(self):
        train_raw_data = pd.read_pickle(train_file)
        train_dataset = MovieScriptDataset(train_raw_data)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=10, shuffle=True, collate_fn=my_collate_fn, num_workers=10, drop_last=True)
        return train_loader

    def val_dataloader(self):
        dev_raw_data = pd.read_pickle(dev_file)
        dev_dataset = MovieScriptDataset(dev_raw_data)
        dev_loader = torch.utils.data.DataLoader(
            dev_dataset, batch_size=10, shuffle=False, collate_fn=my_collate_fn, drop_last=True)
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
    label_batch = torch.stack([each_item['label'] for each_item in batch]).float()

    return text_batch, label_batch


if __name__ == "__main__":
    output_size = 10
    input_size = 768
    hidden_size = 200
    training_epochs = 30

    model = LSTM_model(input_size, output_size, hidden_size)

    early_stop_callback = EarlyStopping(
        monitor='avg_val_loss',
        min_delta=0.00,
        patience=15,
        verbose=False,
        mode='min'
    )
    # 3. Init ModelCheckpoint callback, monitoring 'val_loss'
    checkpoint_callback = ModelCheckpoint(
        monitor='avg_val_loss',
        filepath='/home/yzhan273/Research/MPAA/Severity_Class_Pred/rank/LT_save_model_earlystop_lambda',
        mode='min')


    trainer = pl.Trainer(fast_dev_run=False, max_epochs=training_epochs, gpus=[6],
                         callbacks=[early_stop_callback],
                        checkpoint_callback=checkpoint_callback)
#     without monitoring
#     trainer = pl.Trainer(fast_dev_run=False, max_epochs=training_epochs, gpus=[6])

    trainer.fit(model)
    
    # save without monitoring
#     torch.save(model.state_dict(), '/home/yzhan273/Research/MPAA/Severity_Class_Pred/rank/LT_save_model_noearlystop_lambda')

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