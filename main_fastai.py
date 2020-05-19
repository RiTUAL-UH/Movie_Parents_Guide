#%%
import pandas as pd
import numpy as np
from textacy import preprocessing
import re

#%%
import warnings
warnings.filterwarnings("ignore")

#%%
from fastai.text import *
from fastai.callbacks.tracker import SaveModelCallback

#%%
torch.manual_seed(1234)

if torch.cuda.is_available():
    print("WARNING: You have a CUDA device")

torch.cuda.set_device('cuda:6')



#%%
doc_list = ['frightening', 'alcohol','nudity', 'violence', 'profanity']
# 0 1 2 3 4 
working_aspect = doc_list[4] 
print("Now working on: ", working_aspect)
print("Current CUDA in use: ", torch.cuda.current_device())
train_dataset='./data/'+ working_aspect +'_all_train.csv'
validation_dataset = './data/'+ working_aspect + '_all_dev.csv'
test_dataset ='./data/'+ working_aspect + '_all_test.csv'

finetune_dataset = './data/id_text_no_aspect.csv'

#%%
df_train = pd.read_csv(train_dataset, sep='\t')
print("Train load finished")
df_dev = pd.read_csv(validation_dataset, sep='\t')
print("Dev load finished")
df_test = pd.read_csv(test_dataset, sep='\t')
print("Test load finished")
# df_finetune = pd.read_csv(finetune_dataset, sep='\t')
# print("Fine tune load finished")

# %%
df_train["Aspect_rating"].unique()

# %%
def fix_cmt(x:str):
    x = re.sub(u"[\uFFFD_]+", " ", x)
    x = preprocessing.normalize.normalize_whitespace(x)
    x = preprocessing.normalize.normalize_hyphenated_words(x)
    x = preprocessing.replace.replace_emails(x)
    x = preprocessing.replace.replace_hashtags(x)
    x = preprocessing.replace.replace_numbers(x)
    x = preprocessing.replace.replace_urls(x)
    return x

# %%
df_train["text"] = df_train["text"].apply(fix_cmt)
print("Train preprocessing finished")
df_dev["text"] = df_dev["text"].apply(fix_cmt)
print("Dev preprocessing finished")
df_test["text"] = df_test["text"].apply(fix_cmt)
print("Test preprocessing finished")
# df_finetune["text"] = df_finetune["text"].apply(fix_cmt)

# %%  LM TRAIN ===============================================================
df_lm_train = pd.concat( [df_train, df_dev] )
df_lm_train.shape

# %%
severity = TextLMDataBunch.from_df(".", train_df=df_lm_train, valid_df=df_test, text_cols=1, label_cols=7)
# no_aspect = TextLMDataBunch.from_df(".", train_df=df_finetune, valid_df=df_dev, text_cols=1)
# %%
severity.show_batch()

# %%
learn = language_model_learner(severity, Transformer, drop_mult=0.5)
learn.unfreeze()
learn.fit_one_cycle(10)

# %%
learn.save('lm_databunch'+'_'+ working_aspect + '_.save')
learn.save_encoder('lm_enc'+'_'+ working_aspect + '_.save')

# %%  CLAS =====================================================================
clsdb = TextClasDataBunch.from_df(".",
                            train_df=df_train,
                            valid_df=df_dev,
                            test_df=df_test,
                            text_cols=1,
                            label_cols=7,
                            vocab=severity.train_ds.vocab,
                            bs=40)

# %%
clsdb.show_batch()

# %%
learn = text_classifier_learner(clsdb, Transformer, drop_mult=0.5)

#%%
learn.load_encoder('lm_enc'+'_'+ working_aspect + '_.save')
learn.fit_one_cycle(4)

#%%
learn.save('stage1-clas'+'_'+ working_aspect)

#%%
learn.lr_find(num_it=400, end_lr=0.1, stop_div=False)

#%%
learn.recorder.plot()

#%%
learn.load('stage1-clas'+'_'+ working_aspect)

#%%
learn.unfreeze()

#%%
learn.callback_fns

#%%
learn.callback_fns = [ learn.callback_fns[0] ]
learn.callback_fns += [ partial(SaveModelCallback, name='beer_model'+'_'+ working_aspect + '_.save') ]

#%%
learn.fit_one_cycle(13, slice(5e-5,5e-3))

#%%
learn.recorder.plot_losses()

#%%
learn.save('beer_model'+'_'+ working_aspect + '_.save')
learn.load('beer_model'+'_'+ working_aspect + '_.save')

#%%
from sklearn.metrics import classification_report

#%%
i2c = {i:c for c,i in learn.data.c2i.items()}

# %%
y_pred = torch.argmax(learn.get_preds()[0], dim=1).numpy()
y_pred_str = [i2c[i] for i in y_pred]

#%%
report = classification_report(df_dev["Aspect_rating"], y_pred_str, digits = 4)
print(report)

#%%
y_pred = torch.argmax(learn.get_preds(DatasetType.Test)[0], dim=1).numpy()
y_pred_str = [i2c[i] for i in y_pred]

#%%
report_test = classification_report(df_test["Aspect_rating"], y_pred_str, digits = 4)
print(report_test)

# %%
# df_result = pd.DataFrame(data={"ID":df_test["ID"], "Label":y_pred_str})

# %%
# df_result.to_csv("result.csv", index=False)

# %%