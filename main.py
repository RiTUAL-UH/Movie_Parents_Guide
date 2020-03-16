import os
import time
import itertools
import load_data
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from models.LSTM import LSTMClassifier
from sklearn.metrics import classification_report, f1_score  

if torch.cuda.is_available():
    print("WARNING: You have a CUDA device")

device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter = load_data.load_dataset()

def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)
    
def train_model(model, train_iter, epoch):
    total_epoch_loss = 0
    total_epoch_acc = 0
    total_prediction_digits = []
    total_target_digits = []
    model.cuda()
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    steps = 0
    model.train()
    for idx, batch in enumerate(train_iter):
        text = batch.text[0]
        target = batch.label
        target = torch.autograd.Variable(target).long()
        if torch.cuda.is_available():
            text = text.cuda()
            target = target.cuda()
        # One of the batch returned by BucketIterator has length different than 40 (batch size).
        if (text.size()[0] is not 40):
            continue
        optim.zero_grad()
        prediction = model(text)
        loss = loss_fn(prediction, target)

        prediction_digits = torch.max(prediction, 1)[1].view(target.size()).data.cpu().numpy()
        target_digits = target.data.cpu().numpy()
        
        total_prediction_digits.append(prediction_digits.tolist())
        total_target_digits.append(target_digits.tolist())
        
        num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
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
                                   average='macro')
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
            if (text.size()[0] is not 40):
                continue
            target = batch.label
            target = torch.autograd.Variable(target).long()
            if torch.cuda.is_available():
                text = text.cuda()
                target = target.cuda()
            prediction = model(text)
                 
            loss = loss_fn(prediction, target)
            
            prediction_digits = torch.max(prediction, 1)[1].view(target.size()).data.cpu().numpy()
            target_digits = target.data.cpu().numpy()

            total_prediction_digits.append(prediction_digits.tolist())
            total_target_digits.append(target_digits.tolist())
        
            num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
            acc = 100.0 * num_corrects/len(batch)
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

    val_f1_score = f1_score(list(itertools.chain(*total_target_digits)),
                                   list(itertools.chain(*total_prediction_digits)), 
                                   average='macro')        
    return total_epoch_loss/len(val_iter), total_epoch_acc/len(val_iter), val_f1_score
	

learning_rate = 2e-5
batch_size = 8
output_size = 4
hidden_size = 256
embedding_length = 300
training_epochs = 20

model = LSTMClassifier(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
loss_fn = F.cross_entropy

for epoch in range(training_epochs):
    train_loss, train_acc, train_f1 = train_model(model, train_iter, epoch)
    val_loss, val_acc, val_f1 = eval_model(model, valid_iter)
    
    print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Train F1: {train_f1:.4f}, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}')
    
test_loss, test_acc, test_f1 = eval_model(model, test_iter)
print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%, Test F1: {test_f1:.4f}')

''' Let us now predict the sentiment on a single sentence just for the testing purpose. '''
test_sen1 = "This is one of the best creation of Nolan. I can say, it's his magnum opus. Loved the soundtrack and especially those creative dialogues."
test_sen2 = "Ohh, such a ridiculous movie. Not gonna recommend it to anyone. Complete waste of time and money."

test_sen1 = TEXT.preprocess(test_sen1)
test_sen1 = [[TEXT.vocab.stoi[x] for x in test_sen1]]

test_sen2 = TEXT.preprocess(test_sen2)
test_sen2 = [[TEXT.vocab.stoi[x] for x in test_sen2]]

test_sen = np.asarray(test_sen1)
test_sen = torch.LongTensor(test_sen)
test_tensor = Variable(test_sen, volatile=True)
test_tensor = test_tensor.cuda()
model.eval()
output = model(test_tensor, 1)
out = F.softmax(output, 1)
if (torch.argmax(out[0]) == 1):
    print ("Sentiment: Positive")
else:
    print ("Sentiment: Negative")
