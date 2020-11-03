from pytorch_metric_learning import losses, miners, distances, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import pickle
import pandas as pd

doc_list = ['frightening', 'alcohol', 'nudity', 'violence', 'profanity']
working_aspect = doc_list[0]
print('Now working on:', working_aspect)
base_dir = '/home/yzhan273/Research/MPAA/Severity_Class_Pred/sent_bert/data_sent_emb/'
train_file = base_dir + working_aspect + '_train.pkl'
dev_file = base_dir + working_aspect + '_dev.pkl'


device = torch.device("cuda:5")

class LSTM_model(nn.Module):
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
        # self.label = nn.Linear(self.hidden_size * self.direction, self.output_size)

    def forward(self, input_sentence, batch_size=None):
        # print("here")
        if batch_size is None:
            # Initial hidden state of the LSTM (num_layers * num_directions, batch, hidden_size)
            h_0 = torch.zeros(1 * self.direction, self.batch_size, self.hidden_size).requires_grad_().to(
                device='cuda:5')
            # Initial cell state of the LSTM
            c_0 = torch.zeros(1 * self.direction, self.batch_size, self.hidden_size).requires_grad_().to(
                device='cuda:5')

        else:
            h_0 = torch.zeros(1 * self.direction, batch_size, self.hidden_size).requires_grad_().to(device='cuda:5')
            c_0 = torch.zeros(1 * self.direction, batch_size, self.hidden_size).requires_grad_().to(device='cuda:5')

        output, (final_hidden_state, final_cell_state) = self.lstm(input_sentence, (h_0, c_0))

        output = pad_packed_sequence(output, batch_first=True)  # padded seq, lengths
        output = torch.max(output[0], dim=1)[0]  # after max, (max tensor, max_indices)

        # final_output = self.label(output)

        return output


def train(model, loss_func, mining_func, device, train_loader, optimizer, epoch):
    model.train()
    loss_collect = []
    for batch_idx, batch in enumerate(train_loader):
        text, text_len, labels = batch
        text, labels = text.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(text, len(text_len))
        indices_tuple = mining_func(embeddings, labels)
        loss = loss_func(embeddings, labels, indices_tuple)
        loss.backward()
        optimizer.step()

        if batch_idx % 40 == 0:
            print("Epoch {} Iteration {}: Loss = {}, Number of mined triplets = {}".format(epoch, batch_idx, loss, mining_func.num_triplets))
            
        loss_collect.append(loss.item())
        
    print("Epoch loss:", np.mean(loss_collect))
        
        


### convenient function from pytorch-metric-learning ###
def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)


### compute accuracy using AccuracyCalculator from pytorch-metric-learning ###
def test(dataset, model, accuracy_calculator):
    embeddings, labels = get_all_embeddings(dataset, model)
    print("Computing accuracy")
    accuracies = accuracy_calculator.get_accuracy(embeddings,
                                                  embeddings,
                                                  np.squeeze(labels),
                                                  np.squeeze(labels),
                                                  True)
    print("Test set accuracy (MAP@10) = {}".format(accuracies["mean_average_precision_at_r"]))



# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,), (0.3081,))
# ])
#
# batch_size = 256


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


train_raw_data = pd.read_pickle(train_file)
train_dataset = MovieScriptDataset(train_raw_data)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=20, shuffle=True, collate_fn=my_collate_fn, num_workers=10)

dev_raw_data = pd.read_pickle(dev_file)
dev_dataset = MovieScriptDataset(dev_raw_data)
dev_loader = torch.utils.data.DataLoader(
    dev_dataset, batch_size=1, shuffle=False, collate_fn=my_collate_fn)

# dataset1 = datasets.MNIST('.', train=True, download=True, transform=transform)
# dataset2 = datasets.MNIST('.', train=False, transform=transform)
#
# train_loader = torch.utils.data.DataLoader(dataset1, batch_size=256, shuffle=True)
# test_loader = torch.utils.data.DataLoader(dataset2, batch_size=256)

output_size = 4
input_size = 768
hidden_size = 200
training_epochs = 30

model = LSTM_model(input_size, output_size, hidden_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 40

### pytorch-metric-learning stuff ###
distance = distances.LpDistance()
reducer = reducers.MeanReducer()
loss_func = losses.ProxyNCALoss(output_size, hidden_size * 2, softmax_scale=1)
mining_func = miners.TripletMarginMiner(margin=0.2, distance=distance, type_of_triplets="semihard")
accuracy_calculator = AccuracyCalculator(include=("mean_average_precision_at_r",), k=10)
### pytorch-metric-learning stuff ###


for epoch in range(1, num_epochs + 1):
    train(model, loss_func, mining_func, device, train_loader, optimizer, epoch)
    # test(dataset2, model, accuracy_calculator)

    
torch.save(model.state_dict(), './metric_saved_model_'+ working_aspect +'.ckpt')