import os
import time
import itertools
import load_data
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np

# from models.LSTM_pooling import LSTMClassifier
from models.hierarchical_att_model import HierAttNet



# from spacecutter.losses import CumulativeLinkLoss

from sklearn.metrics import classification_report, f1_score

torch.manual_seed(1234)

if torch.cuda.is_available():
    print("WARNING: You have a CUDA device")

device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

max_sent_length = 100
max_word_length = 100
batch_size = 40


TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter = load_data.load_dataset()

print("Current CUDA in use: ", torch.cuda.current_device())


# stringlist = [none, mild, moderate, severe]
# [['5', '4', '9', '1', '0', '0', '0', '0', '0', '1'], 
#  ['9', '16', '26', '1', '1', '3', '1', '2', '0', '3'],
#  ['0', '0', '0', '12', '0', '0', '1', '0', '0', '3'],
#  ['0', '6', '2', '2', '0', '0', '5', '0', '5', '0']]
def severity_to_distribution(stringlist):
    intlist = [list(map(int, one_stringlist)) for one_stringlist in stringlist]
    distribution_tensor = torch.t(torch.Tensor(intlist).long())
#     print(distribution_tensor)
    # add small value tackle 0 in distribution
    distribution_tensor = torch.add(distribution_tensor, 0.0001)
#     print(distribution_tensor)
    distribution_sum = torch.sum(distribution_tensor, 1).view(-1,1).float()
#     print(distribution_sum)
    distribution_tensor = torch.div(distribution_tensor, distribution_sum)
#     print(distribution_tensor)
    return distribution_tensor

def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)
    
def train_model(model, train_iter, epoch):
    total_epoch_loss = 0
    total_epoch_acc = 0
    total_prediction_digits = []
    total_target_digits = []
    model.to(device)
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = 5e-4)
    steps = 0
    model.train()
    for idx, batch in enumerate(train_iter):
        text = batch.text[0]
#         print(text)
#         print(text.shape)
        text = text.view(-1, max_sent_length, max_word_length)
        
        # learning target is distribution
        target = severity_to_distribution(
            [batch.None1, batch.Mild, batch.Moderate, batch.Severe]
        )
        
#         target = severity_to_distribution(
#             [batch.None1, batch.Mild, batch.Moderate, batch.Severe]
#         )
        
#         print(target)
        
        target_digits = torch.Tensor([int(each) for each in batch.Aspect_rating]).long()
#         print(target_digits)
#         time.sleep(1)
        
        target = torch.autograd.Variable(target)
        
        if torch.cuda.is_available():
            text = text.to(device)
            target = target.to(device)
        # One of the batch returned by BucketIterator has length different than 40 (batch size).
        if (text.size()[0] is not batch_size):
            continue
        
#         print("target: ", target)
        optim.zero_grad()
        model._init_hidden_state()
        
        # predict distribution should be a log probability
        prediction = F.softmax(model(text))
#         print("here!")
#         print("prediction:", prediction)

        # loss = loss_fn(prediction.log(), target) 
        loss = loss_fn(prediction.log(), target) + loss_fn(target.log(), prediction)
        
#         print("loss: ",loss)

        prediction_digits = torch.max(prediction, 1)[1]
#         print("prediction_digits: ", prediction_digits)
        prediction_digits = prediction_digits.view(target_digits.size()).data.cpu().numpy()
#         print("prediction_digits: ", prediction_digits)
        target_digits = target_digits.data.cpu().numpy()
#         print("target_digits:", target_digits)
        
        total_prediction_digits.append(prediction_digits.tolist())
#         print("total_prediction_digits: ", total_prediction_digits)
        
        total_target_digits.append(target_digits.tolist())
#         print("total_target_digits:", total_target_digits)
        num_corrects = prediction_digits == target_digits
#         num_corrects = torch.max(prediction, 1)[1].view(target_digits.size()).data == target_digits.data
        
#         print("num_corrects: ", num_corrects)
        num_corrects = (num_corrects).astype(int).sum()
        
        acc = 100.0 * num_corrects/len(batch)
        
        loss.backward()
        clip_gradient(model, 1e-1)
        optim.step()
        steps += 1
        
        if steps % 100 == 0:
            print (f'Epoch: {epoch+1}, Idx: {idx+1}, Training Loss: {loss.item():.4f}, Training Accuracy: {acc.item(): .2f}%')
        
        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()
    
    train_f1_score = f1_score(list(itertools.chain(*total_target_digits)),
                                   list(itertools.chain(*total_prediction_digits)), 
                                   average='weighted')
    return total_epoch_loss/len(train_iter), total_epoch_acc/len(train_iter), train_f1_score


def eval_model(model, val_iter):
    total_epoch_loss = 0
    total_epoch_acc = 0
    total_prediction_digits = []
    total_target_digits = []
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(val_iter):
            text = batch.text[0]
            text = text.view(-1, max_sent_length, max_word_length)
            target = F.softmax(severity_to_distribution(
            [batch.None1, batch.Mild, batch.Moderate, batch.Severe]
            ), dim =1)
        
#             print(target)
#             time.sleep(3)
            
            if (text.size()[0] is not batch_size):
                continue
                

            target_digits = torch.Tensor([int(each) for each in batch.Aspect_rating]).long()
            
            target = torch.autograd.Variable(target)
            
            if torch.cuda.is_available():
                text = text.to(device)
                target = target.to(device)
 
            prediction =  F.softmax(model(text),dim =1)
                 
            loss = loss_fn(prediction.log(), target) + loss_fn(target.log(), prediction)
            
            prediction_digits = torch.max(prediction, 1)[1]
#             print("prediction_digits: ", prediction_digits)
            prediction_digits = prediction_digits.view(target_digits.size()).data.cpu().numpy()
#             print("prediction_digits: ", prediction_digits)
            target_digits = target_digits.data.cpu().numpy()
#             print("target_digits:", target_digits)

            total_prediction_digits.append(prediction_digits.tolist())
#             print("total_prediction_digits: ", total_prediction_digits)

            total_target_digits.append(target_digits.tolist())
#             print("total_target_digits:", total_target_digits)
            num_corrects = prediction_digits == target_digits


#             print("num_corrects: ", num_corrects)
            num_corrects = (num_corrects).astype(int).sum()
            
            acc = 100.0 * num_corrects/len(batch)
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

    val_f1_score = f1_score(list(itertools.chain(*total_target_digits)),
                                   list(itertools.chain(*total_prediction_digits)), 
                                   average='weighted')        
    return total_epoch_loss/len(val_iter), total_epoch_acc/len(val_iter), val_f1_score
	

learning_rate = 5e-3

output_size = 4
hidden_size = 100
sent_hidden_size = 50
word_hidden_size = 50
embedding_length = 300
training_epochs = 50


# model = LSTMClassifier(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)

model = HierAttNet(word_hidden_size, sent_hidden_size, vocab_size, embedding_length, batch_size, output_size, word_embeddings, max_sent_length, max_word_length)

# model = LSTMClassifier(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
loss_fn = F.kl_div

for epoch in range(training_epochs):
    train_loss, train_acc, train_f1 = train_model(model, train_iter, epoch)
    val_loss, val_acc, val_f1 = eval_model(model, valid_iter)
    
    print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Train F1: {train_f1:.4f}, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}')
    
test_loss, test_acc, test_f1 = eval_model(model, test_iter)
print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%, Test F1: {test_f1:.4f}')


