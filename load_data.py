# _*_ coding: utf-8 _*_

import os
import sys
import torch
from torch.nn import functional as F
import numpy as np
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe

def load_dataset(test_sen=None):

    """
    tokenizer : Breaks sentences into a list of words. If sequential=False, no tokenization is applied
    Field : A class that stores information about the way of preprocessing
    fix_length : An important property of TorchText is that we can let the input to be variable length, and TorchText will
                 dynamically pad each sequence to the longest sequence in that "batch". But here we are using fi_length which
                 will pad each sequence to have a fix length of 200.
                 
    build_vocab : It will first make a vocabulary or dictionary mapping all the unique words present in the train_data to an
                  idx and then after it will use GloVe word embedding to map the index to the corresponding word embedding.
                  
    vocab.vectors : This returns a torch tensor of shape (vocab_size x embedding_dim) containing the pre-trained word embeddings.
    BucketIterator : Defines an iterator that batches examples of similar lengths together to minimize the amount of padding needed.
    
    """
    
    doc_list = ['frightening', 'alcohol','nudity', 'violence', 'profanity']
    # 0 1 2 3 4 
    working_aspect = doc_list[3] 
    print("Now working on: ", working_aspect)
    
    tokenize = lambda x: x.split()
    TEXT = data.Field(sequential=True, 
                      tokenize=tokenize, 
                      lower=True, 
                      include_lengths=True, 
                      batch_first=True, 
                      fix_length=10000,
                      pad_first=True)   
    
    LABEL = data.RawField()
#     LABEL = data.LabelField()
    
    NONE1 = data.RawField()
    MILD = data.RawField()
    MODERATE = data.RawField()
    SEVERE = data.RawField()
    
    
    # id	text	None	Mild	Moderate	Severe	MPAA_Rating	Aspect_rating
    fields = [("id", None),
                     ("text", TEXT), 
                     ("None1", NONE1),
                     ("Mild", MILD),
                     ("Moderate", MODERATE),
                     ("Severe", SEVERE), 
                     ("MPAA_Rating", None),
                     ("Aspect_rating", LABEL)]
    

    train_data, valid_data, test_data = data.TabularDataset.splits(path='./', 
                                            format='csv', 
                                            train='./data/'+ working_aspect +'_all_train.csv', 
                                            validation = './data/'+ working_aspect + '_all_dev.csv',
                                            test='./data/'+ working_aspect + '_all_test.csv',
                                            fields=fields, 
                                            skip_header=True,
                                            csv_reader_params = {'delimiter':'\t'} )
    
    # test fast
#     train_data, valid_data, test_data = data.TabularDataset.splits(path='./', 
#                                             format='csv', 
#                                             train='./data/test_fast.csv', 
#                                             validation = './data/test_fast.csv',
#                                             test='./data/test_fast.csv',
#                                             fields=fields, 
#                                             skip_header=True,
#                                             csv_reader_params = {'delimiter':'\t'} )
    
   
#     print("Now working on:")
#     TEXT.build_vocab(train_data, vectors=GloVe(name='6B', dim=300))

    TEXT.build_vocab(train_data, valid_data, test_data, vectors=GloVe(name='840B', dim=300))
#     LABEL.build_vocab(train_data)


    word_embeddings = TEXT.vocab.vectors
    print ("Length of Text Vocabulary: " + str(len(TEXT.vocab)))
    print ("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
    
#     print(TEXT.vocab.stoi)

    
    train_iter, valid_iter, test_iter = data.BucketIterator.splits((train_data, valid_data, test_data), batch_size=40, sort_key=lambda x: len(x.text), repeat=False, shuffle=False)


    vocab_size = len(TEXT.vocab)

    return TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter
