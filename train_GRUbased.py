import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import pickle
import glob
from tqdm import tqdm

# ========================
# constant
# ========================
DEFOG_META_PATH = "../data/defog_metadata.csv"
DEFOG_FOLDER = "../data/train/defog/*.csv"

# ========================
# settings
# ========================
fe = "047"
if not os.path.exists(f"../output/fe/fe{fe}"):
    os.makedirs(f"../output/fe/fe{fe}")
    os.makedirs(f"../output/fe/fe{fe}/save")

meta = pd.read_parquet("../output/fe/fe039/fe039_defog_meta.parquet")

cols = ["AccV","AccML","AccAP"]
num_cols = ["AccV","AccML","AccAP",'AccV_lag_diff', 'AccV_lead_diff', 'AccML_lag_diff', 'AccML_lead_diff',
       'AccAP_lag_diff', 'AccAP_lead_diff']
target_cols = ["StartHesitation","Turn","Walking"]
seq_len = 5000
shift = 2500
offset = 1250

num_array = []
target_array = []
subject_list = []
valid_array = []
id_list = []
mask_array = []
pred_use_array = []
time_array = []
d_list = []

data_list = glob.glob(DEFOG_FOLDER)

for i,s in tqdm(zip(meta["Id"].values,
               meta["sub_id"].values)):
    path = f"../data/train/defog\\{i}.csv"
    if path in data_list:
        d_list.append(1)
        df = pd.read_csv(path)
        # label shift
        df["Event"]=df[["StartHesitation","Turn","Walking"]].max(axis=1)
        arr=df[["Event","Valid"]].astype(int).values
        # length = len(arr)
        sum_event=arr[:,0].sum()
        acc=[]
        for i in range(len(arr[:,0])//2):
            shifted_arr = np.roll(arr[:,0], -i)
            shifted_arr[-(i + 1):] = 0
            arr2=(shifted_arr*arr[:,1]).sum()
            acc.append(arr2/sum_event)
            ind=np.argmax(acc)
            if acc[-1]==1:
                break
        df[["StartHesitation","Turn","Walking"]]=df[["StartHesitation","Turn","Walking"]].shift(-ind).fillna(0)
        df["valid"] = df["Valid"] & df["Task"]
        df["valid"] = df["valid"].astype(int)
        batch = (len(df)-1) // shift
        for c in cols:
            df[f"{c}_lag_diff"] = df[c].diff()
            df[f"{c}_lead_diff"] = df[c].diff(-1)
        
        sc = StandardScaler()
        df[num_cols] = sc.fit_transform(df[num_cols].values)
        df[num_cols] = df[num_cols].fillna(0)
        
        num = df[num_cols].values
        target = df[target_cols].values
        valid = df["valid"].values
        time = df["Time"].values
        num_array_ = np.zeros([batch,seq_len,9])
        target_array_ = np.zeros([batch,seq_len,3])
        time_array_ = np.zeros([batch,seq_len],dtype=int)
        mask_array_ = np.zeros([batch,seq_len],dtype=int)
        pred_use_array_ = np.zeros([batch,seq_len],dtype=int)
        valid_array_ = np.zeros([batch,seq_len],dtype=int)
        for n,b in enumerate(range(batch)):
            if b == (batch - 1):
                num_ = num[b*shift : ]
                num_array_[b,:len(num_),:] = num_
                target_ = target[b*shift : ]
                target_array_[b,:len(target_),:] = target_
                mask_array_[b,:len(target_)] = 1
                pred_use_array_[b,offset:len(target_)] = 1
                time_ = time[b*shift : ]
                time_array_[b,:len(time_)] = time_
                valid_ = valid[b*shift : ]
                valid_array_[b,:len(valid_)] = valid_
            elif b == 0:
                num_ = num[b*shift:b*shift+seq_len]
                num_array_[b,:,:] = num_
                target_ = target[b*shift:b*shift + seq_len]
                target_array_[b,:,:] = target_
                mask_array_[b,:] = 1
                pred_use_array_[b,:shift + offset] = 1
                time_ = time[b*shift:b*shift + seq_len]
                time_array_[b,:] = time_
                valid_ = valid[b*shift:b*shift + seq_len]
                valid_array_[b,:] = valid_
            else:
                num_ = num[b*shift:b*shift+seq_len]
                num_array_[b,:,:] = num_
                target_ = target[b*shift:b*shift + seq_len]
                target_array_[b,:,:] = target_
                mask_array_[b,:] = 1
                pred_use_array_[b,offset:shift + offset] = 1
                time_ = time[b*shift:b*shift + seq_len]
                time_array_[b,:] = time_
                valid_ = valid[b*shift:b*shift + seq_len]
                valid_array_[b,:] = valid_

        num_array.append(num_array_)
        target_array.append(target_array_)
        mask_array.append(mask_array_)
        pred_use_array.append(pred_use_array_)
        time_array.append(time_array_)
        valid_array.append(valid_array_)
        subject_list += [s for _ in range(batch)]
        id_list += [i for _ in range(batch)] 
    else:
        d_list.append(0)

num_array = np.concatenate(num_array,axis=0)
target_array =np.concatenate(target_array,axis=0)
mask_array =  np.concatenate(mask_array,axis=0)
pred_use_array = np.concatenate(pred_use_array,axis=0)
time_array = np.concatenate(time_array,axis=0)
valid_array = np.concatenate(valid_array,axis=0)

df_id = pd.DataFrame()
df_id["Id"] = id_list
df_id["subject"] = subject_list

np.save(f"../output/fe/fe{fe}/fe{fe}_num_array.npy",num_array)
np.save(f"../output/fe/fe{fe}/fe{fe}_target_array.npy",target_array)
np.save(f"../output/fe/fe{fe}/fe{fe}_mask_array.npy",mask_array)
np.save(f"../output/fe/fe{fe}/fe{fe}_time_array.npy",time_array)
np.save(f"../output/fe/fe{fe}/fe{fe}_pred_use_array.npy",pred_use_array)
np.save(f"../output/fe/fe{fe}/fe{fe}_valid_array.npy",valid_array)

df_id.to_parquet(f"../output/fe/fe{fe}/fe{fe}_id.parquet")

import warnings
warnings.simplefilter('ignore')
import math
import pandas as pd
import numpy as np
import sys
import time
import datetime
from contextlib import contextmanager
import logging
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score,average_precision_score
from sklearn.model_selection import StratifiedKFold, KFold,GroupKFold,StratifiedGroupKFold
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder
from torch.nn import TransformerDecoder
from torch.nn import LayerNorm
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.optim import lr_scheduler
from transformers import AdamW, get_linear_schedule_with_warmup
import gc
import random
import os
sys.path.append("../src/")
# from logger import setup_logger, LOGGER
# from util_tool import reduce_mem_usage
pd.set_option('display.max_columns', 300)

logger = logging.getLogger("logger") 

def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() 
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() 
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

debug = False
ex = "153"
if not os.path.exists(f"../output/exp/ex{ex}"):
    os.makedirs(f"../output/exp/ex{ex}")
    os.makedirs(f"../output/exp/ex{ex}/ex{ex}_model")
logger_path = f"../output/exp/ex{ex}/ex_{ex}.txt"
model_path =f"../output/ex{ex}/ex{ex}.pth"
# config
seed = 0
shuffle = True
n_splits = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model config
batch_size = 24
n_epochs = 15
lr = 1e-3
weight_decay = 0.05
num_warmup_steps = 10

id_path = f"../output/fe/fe047/fe047_id.parquet"
numerical_path = f"../output/fe/fe047/fe047_num_array.npy"
target_path = f"../output/fe/fe047/fe047_target_array.npy"
mask_path = f"../output/fe/fe047/fe047_mask_array.npy"
valid_path = f"../output/fe/fe047/fe047_valid_array.npy"
pred_use_path = f"../output/fe/fe047/fe047_pred_use_array.npy"

def preprocess(numerical_array, 
               mask_array,
               valid_array,
               ):
    
    attention_mask = mask_array == 0

    return {
        'input_data_numerical_array': numerical_array,
        'input_data_mask_array': mask_array,
        'input_data_valid_array': valid_array,
        'attention_mask': attention_mask,
    }

class FogDataset(Dataset):
    def __init__(self, numerical_array, 
                 mask_array,valid_array,
                 train = True, y = None):
        self.numerical_array = numerical_array
        self.mask_array = mask_array
        self.valid_array = valid_array
        self.train = train
        self.y = y
    
    def __len__(self):
        return len(self.numerical_array)

    def __getitem__(self, item):
        data = preprocess(
            self.numerical_array[item],
            self.mask_array[item],
            self.valid_array[item], 
            
        )

        # Return the processed data where the lists are converted to `torch.tensor`s
        if self.train : 
            return {
              'input_data_numerical_array': torch.tensor(data['input_data_numerical_array'],dtype=torch.float32),
              'input_data_mask_array':torch.tensor(data['input_data_mask_array'], dtype=torch.long),  
              'input_data_valid_array':torch.tensor(data['input_data_valid_array'], dtype=torch.long),   
              'attention_mask': torch.tensor(data["attention_mask"], dtype=torch.bool),
              "y":torch.tensor(self.y[item], dtype=torch.float32)
               }
        else:
            return {
             'input_data_numerical_array': torch.tensor(data['input_data_numerical_array'],dtype=torch.float32),
              'input_data_mask_array':torch.tensor(data['input_data_mask_array'], dtype=torch.long),  
              'input_data_valid_array':torch.tensor(data['input_data_valid_array'], dtype=torch.long),  
              'attention_mask': torch.tensor(data["attention_mask"], dtype=torch.bool),
               }
        
class FogRnnModel(nn.Module):
    def __init__(
        self, dropout=0.2,
        input_numerical_size=9,
        numeraical_linear_size = 64,
        model_size = 128,
        linear_out = 128,
        out_size=3):
        super(FogRnnModel, self).__init__()
        self.numerical_linear  = nn.Sequential(
                nn.Linear(input_numerical_size, numeraical_linear_size),
                nn.LayerNorm(numeraical_linear_size)
            )
        
        self.rnn = nn.GRU(numeraical_linear_size, model_size,
                            num_layers = 2, 
                            batch_first=True,
                            bidirectional=True)
        self.linear_out  = nn.Sequential(
                nn.Linear(model_size*2, 
                          linear_out),
                nn.LayerNorm(linear_out),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(linear_out, 
                          out_size))
        self._reinitialize()
        
    def _reinitialize(self):
        """
        Tensorflow/Keras-like initialization
        """
        for name, p in self.named_parameters():
            if 'rnn' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(p.data)
                elif 'bias_ih' in name:
                    p.data.fill_(0)
                    # Set forget-gate bias to 1
                    n = p.size(0)
                    p.data[(n // 4):(n // 2)].fill_(1)
                elif 'bias_hh' in name:
                    p.data.fill_(0)
    
    def forward(self, numerical_array,
                mask_array,
                attention_mask):
        
        numerical_embedding = self.numerical_linear(numerical_array)
        output,_ = self.rnn(numerical_embedding)
        output = self.linear_out(output)
        return output
    
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def timer(name):
    t0 = time.time()
    yield 
    logger.info(f'[{name}] done in {time.time() - t0:.0f} s')

df_id = pd.read_parquet(id_path)
numerical_array = np.load(numerical_path)
target_array = np.load(target_path)
mask_array = np.load(mask_path)
valid_array = np.load(valid_path)
pred_use_array = np.load(pred_use_path)

target1 = []
target2 = []
target3 = []
for i in range(len(target_array)):
    target1.append(np.sum(target_array[i,:,0]))
    target2.append(np.sum(target_array[i,:,1]))
    target3.append(np.sum(target_array[i,:,2]))

df_id["target1"] = target1
df_id["target2"] = target2
df_id["target3"] = target3
df_id["target1_1"] = df_id["target1"] > 0
df_id["target2_1"] = df_id["target2"] > 0
df_id["target3_1"] = df_id["target3"] > 0
df_id["target1_1"] = df_id["target1_1"].astype(np.int)
df_id["target2_1"] = df_id["target2_1"].astype(np.int)
df_id["target3_1"] = df_id["target3_1"].astype(np.int)

df_id["group"] = 0
df_id.loc[df_id["target1_1"] > 0,"group"] = 1
df_id.loc[df_id["target2_1"] > 0,"group"] = 2
df_id.loc[df_id["target3_1"] > 0,"group"] = 3

df_id["group"].value_counts()

with timer("gru"):
    set_seed(seed)
    y_oof = np.empty([len(target_array),5000,3])
    gkf = StratifiedGroupKFold(n_splits=n_splits,shuffle=True,random_state = seed)
    for fold, (train_idx, valid_idx) in enumerate(gkf.split(numerical_array, 
                                                             y = df_id["group"].values,
                                                             groups=df_id["subject"].values)):
        logger.info(f"start fold:{fold}")
        with timer(f"fold {fold}"):
            train_numerical_array = numerical_array[train_idx]
            train_target_array = target_array[train_idx]
            train_mask_array = mask_array[train_idx]
            train_valid_array = valid_array[train_idx]

            val_numerical_array = numerical_array[valid_idx]
            val_target_array = target_array[valid_idx]
            val_mask_array = mask_array[valid_idx]
            val_valid_array = valid_array[valid_idx]
            val_pred_array = pred_use_array[valid_idx]
            
            train_ = FogDataset(train_numerical_array,
                                train_mask_array,
                                train_valid_array, 
                                train=True,
                                y=train_target_array)
            
            val_ = FogDataset(val_numerical_array,
                                val_mask_array,
                                val_valid_array, 
                                train=True,
                                y=val_target_array)
            
            
            train_loader = DataLoader(dataset=train_, 
                                  batch_size=batch_size, 
                                  shuffle = True , num_workers=8)
            val_loader = DataLoader(dataset=val_, 
                                batch_size=batch_size, 
                                shuffle = False , num_workers=8)
            
            model = FogRnnModel()
            model = model.to(device)
            param_optimizer = list(model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            optimizer = AdamW(optimizer_grouped_parameters,
                              lr=lr,
                              weight_decay=weight_decay,
                              )
            num_train_optimization_steps = int(len(train_loader) * n_epochs)
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=num_warmup_steps,
                                                        num_training_steps=num_train_optimization_steps)
            criterion = nn.BCEWithLogitsLoss()
            best_val_score = 0
            for epoch in range(n_epochs):
                model.train() 
                train_losses_batch = []
                val_losses_batch = []
                epoch_loss = 0
                train_preds = np.ndarray((0,3))
                tk0 = tqdm(train_loader, total=len(train_loader))
                for d in tk0:
                    # =========================
                    # data loader
                    # =========================
                    input_data_numerical_array = d['input_data_numerical_array'].to(device)
                    input_data_mask_array = d['input_data_mask_array'].to(device)
                    input_data_valid_array = d['input_data_valid_array'].to(device)
                    attention_mask = d['attention_mask'].to(device)
                    y = d["y"].to(device)
                    optimizer.zero_grad()
                    output = model(input_data_numerical_array, 
                                   input_data_mask_array,
                                   attention_mask)
                    loss = criterion(output[(input_data_valid_array == 1)], y[input_data_valid_array == 1])
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    train_losses_batch.append(loss.item())
                train_loss = np.mean(train_losses_batch)
            
                # ==========================
                # eval
                # ==========================
                model.eval()  # switch model to the evaluation mode
                val_preds = np.ndarray((0,5000,3))
                tk0 = tqdm(val_loader, total=len(val_loader))
                with torch.no_grad():  # Do not calculate gradient since we are only predicting
                    # Predicting on validation set
                    for d in tk0:
                        input_data_numerical_array = d['input_data_numerical_array'].to(device)
                        input_data_mask_array = d['input_data_mask_array'].to(device)
                        attention_mask = d['attention_mask'].to(device)
                        output = model(input_data_numerical_array, 
                                   input_data_mask_array,
                                   attention_mask)
                        val_preds = np.concatenate([val_preds, output.sigmoid().detach().cpu().numpy()], axis=0)
                pred_valid_index = (val_mask_array == 1) & (val_pred_array == 1) & (val_valid_array == 1)
                StartHesitation = average_precision_score(val_target_array[pred_valid_index][:,0],
                                                          val_preds[pred_valid_index][:,0])
                Turn = average_precision_score(val_target_array[pred_valid_index][:,1],
                                                          val_preds[pred_valid_index][:,1])
                Walking = average_precision_score(val_target_array[pred_valid_index][:,2],
                                                          val_preds[pred_valid_index][:,2])
                map_score = np.mean([Turn,
                                     Walking])
                logger.info(f"fold:{fold} epoch : {epoch},train loss {train_loss} map:{map_score} starthesi:{StartHesitation} turn:{Turn} walking :{Walking}")
                if map_score >= best_val_score:
                    print("save weight")
                    best_val_score = map_score
                    best_val_preds = val_preds.copy()
                    torch.save(model.state_dict(), f"../output/ex{ex}/{fold}.pth") 
            y_oof[valid_idx] = best_val_preds
    np.save(f"../output/exp/ex{ex}/ex{ex}_oof.npy",y_oof)


val_pred_index = (mask_array == 1) & (pred_use_array == 1) & (valid_array == 1)
StartHesitation = average_precision_score(target_array[val_pred_index][:,0],
                                                          y_oof[val_pred_index ][:,0])
Turn = average_precision_score(target_array[val_pred_index ][:,1],
                                                          y_oof[val_pred_index ][:,1])
Walking = average_precision_score(target_array[val_pred_index ][:,2],
                                                          y_oof[val_pred_index ][:,2])

map_score = np.mean([StartHesitation,
                               Turn,
                               Walking])
LOGGER.info(f"cv map:{map_score} starthesi:{StartHesitation} turn:{Turn} walking :{Walking}")

import json
kaggle_json = {
  "title": f"fog-ex{ex}",
  "id": f"takoihiraokazu/fog-ex{ex}",
  "licenses": [
    {
      "name": "CC0-1.0"
    }
  ]
}

with open(f"../output/exp/ex{ex}/ex{ex}_model/dataset-metadata.json", 'w') as f:
    json.dump(kaggle_json, f)

