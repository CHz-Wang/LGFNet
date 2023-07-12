import os 
import math
import numpy as np 
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy import signal

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import glob
from sklearn.preprocessing import StandardScaler

# Used to collect all test predictions
all_submissions = []

p = '/kaggle/input/tlvmc-parkinsons-freezing-gait-prediction/'
tsfog_ids = [fname.split('.')[0] for fname in os.listdir(p + 'test/tdcsfog')]
defog_ids = [fname.split('.')[0] for fname in os.listdir(p + 'test/defog')]

# Normalization processing function
def sample_normalize(sample):
    mean = tf.math.reduce_mean(sample)
    std = tf.math.reduce_std(sample)
    sample = tf.math.divide_no_nan(sample-mean, std)
    
    return sample.numpy()

# The input sequence is divided into overlapping sequences
def get_blocks(series, columns):
    series = series.copy()
    series = series[columns].values.astype(np.float32)
    
    # Round Up
    block_count = math.ceil(len(series) / CFG['block_size'])

    # Fill in the sequence so that the sequence length can be block_size division means sequence partitioning is executed normally
    series = np.pad(series, pad_width=[[0, block_count*CFG['block_size']-len(series)], [0, 0]])
    
    # Obtain the starting position of sequence partitioning
    block_begins = list(range(0, len(series), CFG['block_stride']))
    block_begins = [x for x in block_begins if x+CFG['block_size'] <= len(series)]
    
    # Sequential partitioning dictionary stored in blocks
    blocks = []
    for begin in block_begins:
        values = series[begin:begin+CFG['block_size']]
        blocks.append({'begin': begin,
                       'end': begin+CFG['block_size'],
                       'values': values})
    
    return blocks


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# tdcsfog
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

# EncoderLayer in transformer
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        
        # multi-head attention layer
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=CFG['fog_model_num_heads'], key_dim=CFG['fog_model_dim'], dropout=CFG['fog_model_mha_dropout'])
        
        # add layer
        self.add = tf.keras.layers.Add()
        
        # normalization layer
        self.layernorm = tf.keras.layers.LayerNormalization()
        
        # FC block
        self.seq = tf.keras.Sequential([tf.keras.layers.Dense(CFG['fog_model_dim'], activation='relu'), # 全连接层
                                        tf.keras.layers.Dropout(CFG['fog_model_encoder_dropout']), 
                                        tf.keras.layers.Dense(CFG['fog_model_dim']), 
                                        tf.keras.layers.Dropout(CFG['fog_model_encoder_dropout']),
                                       ])
    
    def call(self, x):
        attn_output = self.mha(query=x, key=x, value=x)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        x = self.add([x, self.seq(x)])
        x = self.layernorm(x)
        
        return x

# Encoder
class FOGEncoder(tf.keras.Model):
    def __init__(self):
        super().__init__()
        
        self.masking = tf.keras.layers.Masking()
        
        self.first_linear = tf.keras.layers.Dense(CFG['fog_model_dim'])
        
        self.add = tf.keras.layers.Add()
        
        self.first_dropout = tf.keras.layers.Dropout(CFG['fog_model_first_dropout'])
        
        self.enc_layers = [EncoderLayer() for _ in range(CFG['fog_model_num_encoder_layers'])]
        
        self.lstm_layers = [tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(CFG['fog_model_dim'], return_sequences=True)) for _ in range(CFG['fog_model_num_lstm_layers'])]
        
        self.sequence_len = CFG['block_size'] // CFG['patch_size']
        self.pos_encoding = tf.Variable(initial_value=tf.random.normal(shape=(1, self.sequence_len, CFG['fog_model_dim']), stddev=0.02), trainable=True)
        
    def call(self, x, training=None): # (GPU_BATCH_SIZE, CFG['block_size'] // CFG['patch_size'], CFG['patch_size']*3), Example shape (4, 864, 54)
        x = x / 25.0 # Normalization attempt in the segment [-1, 1]
        x = self.masking(x) # Masks a padded timesteps from multi head attention and lstm layers
        x = self.first_linear(x) # (GPU_BATCH_SIZE, CFG['block_size'] // CFG['patch_size'], CFG['fog_model_dim']), Example shape (4, 864, 320)
          
        if training: # augmentation by randomly roll of the position encoding tensor
            random_pos_encoding = tf.roll(tf.tile(self.pos_encoding, multiples=[GPU_BATCH_SIZE, 1, 1]), 
                                          shift=tf.random.uniform(shape=(GPU_BATCH_SIZE,), minval=-self.sequence_len, maxval=0, dtype=tf.int32),
                                          axis=GPU_BATCH_SIZE * [1],
                                          )
            x = self.add([x, random_pos_encoding])
        
        else: # without augmentation 
            x = self.add([x, tf.tile(self.pos_encoding, multiples=[GPU_BATCH_SIZE, 1, 1])])
            
        x = self.first_dropout(x)
        
        km = x._keras_mask # Bug fix (Multi head attention masking does not work on TPU training)
        del x._keras_mask # Bug fix
        for i in range(CFG['fog_model_num_encoder_layers']): x = self.enc_layers[i](x) # (GPU_BATCH_SIZE, CFG['block_size'] // CFG['patch_size'], CFG['fog_model_dim']), Example shape (4, 864, 320)
        x._keras_mask = km # Bug fix
        for i in range(CFG['fog_model_num_lstm_layers']): x = self.lstm_layers[i](x) # (GPU_BATCH_SIZE, CFG['block_size'] // CFG['patch_size'], CFG['fog_model_dim']*2), Example shape (4, 864, 640)
            
        return x

# final model
class FOGModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        
        self.encoder = FOGEncoder()
        self.last_linear = tf.keras.layers.Dense(3) 
        
    def call(self, x): # (GPU_BATCH_SIZE, CFG['block_size'] // CFG['patch_size'], CFG['patch_size']*3), Example shape (4, 864, 54)
        x = self.encoder(x) # (GPU_BATCH_SIZE, CFG['block_size'] // CFG['patch_size'], CFG['fog_model_dim']*2), Example shape (4, 864, 640)
        x = self.last_linear(x) # (GPU_BATCH_SIZE, CFG['block_size'] // CFG['patch_size'], 3), Example shape (4, 864, 3)
        x = tf.nn.sigmoid(x) # Sigmoid activation
        
        return x


'''
============================================================================================================================================================
tdcsfog model_1
============================================================================================================================================================
'''
# config
CFG = {'TPU': 0,
       'block_size': 15552, 
       'block_stride': 15552//32,
       'patch_size': 18, 
       'fog_model_dim': 320,
       'fog_model_num_heads': 6,
       'fog_model_num_encoder_layers': 5,
       'fog_model_num_lstm_layers': 2,
       'fog_model_first_dropout': 0.1,
       'fog_model_encoder_dropout': 0.1,
       'fog_model_mha_dropout': 0.0,
      }

# Train and inference batch size
GPU_BATCH_SIZE = 4
TPU_BATCH_SIZE = GPU_BATCH_SIZE*8

assert CFG['block_size'] % CFG['patch_size'] == 0
assert CFG['block_size'] % CFG['block_stride'] == 0

# get tdcsfog test data and predict
class PredictionFnCallback_tdcs_1(tf.keras.callbacks.Callback):
    
    def __init__(self, prediction_ids, model=None, verbose=0):
        
        if not model is None: self.model = model
        self.verbose = verbose
         
        def init(Id, path):
            series = pd.read_csv(path).reset_index(drop=True)
            series['Id'] = Id
            # low-pass filter
            accv = series.AccV.values
            accml = series.AccML.values
            accap = series.AccAP.values
            wn=70/128
            b,a = signal.butter(8,wn,'low')
            accvft = signal.filtfilt(b,a,accv)
            accmlft = signal.filtfilt(b,a,accml)
            accapft = signal.filtfilt(b,a,accap)
            series['AccV'] = accvft.tolist()
            series['AccML'] = accmlft.tolist()
            series['AccAP'] = accapft.tolist()
            # normalization
            series['AccV'] = sample_normalize(series['AccV'].values)
            series['AccML'] = sample_normalize(series['AccML'].values)
            series['AccAP'] = sample_normalize(series['AccAP'].values)
            
            series_blocks=[]
            for block in get_blocks(series, ['AccV', 'AccML', 'AccAP']): # Example shape (15552, 3)
                # blocks into patches
                values = tf.reshape(block['values'], shape=(CFG['block_size'] // CFG['patch_size'], CFG['patch_size'], 3)) # Example shape (864, 18, 3)
                # patches with features
                values = tf.reshape(values, shape=(CFG['block_size'] // CFG['patch_size'], CFG['patch_size']*3)) # Example shape (864, 54)
                values = tf.expand_dims(values, axis=0) # Example shape (1, 864, 54)
                
                self.blocks.append(values)
                series_blocks.append((self.blocks_counter, block['begin'], block['end']))
                self.blocks_counter += 1
            
            description = {}
            description['series'] = series
            description['series_blocks'] = series_blocks
            self.descriptions.append(description)
            
        self.descriptions = [] # Blocks metadata
        self.blocks = [] # Test data blocks
        self.blocks_counter=0 # Blocks counter
        
        tsfog_ids = prediction_ids
        tsfog_paths = [f'/kaggle/input/tlvmc-parkinsons-freezing-gait-prediction/test/tdcsfog/{tsfog_id}.csv' for tsfog_id in tsfog_ids]
        for tsfog_id, tsfog_path in tqdm(zip(tsfog_ids, tsfog_paths), total=len(tsfog_ids), desc='PredictionFnCallback Initialization', disable=1-verbose): 
            init(tsfog_id, tsfog_path)
            
        self.blocks = tf.concat(self.blocks, axis=0) # Example shape (self.blocks_counter, 864, 54)
        
        # block padding
        self.blocks = tf.pad(self.blocks, 
                             paddings=[[0, math.ceil(self.blocks_counter / (TPU_BATCH_SIZE if CFG['TPU'] else GPU_BATCH_SIZE))*(TPU_BATCH_SIZE if CFG['TPU'] else GPU_BATCH_SIZE)-self.blocks_counter], 
                                                    [0, 0], 
                                                    [0, 0],
                                      ]) # Example shape (self.blocks_counter+pad_value, 864, 54)
        
        print(f'\n[EventPredictionFnCallback Initialization] [Series] {len(self.descriptions)} [Blocks] {self.blocks_counter}\n')
    
    def prediction(self):
        predictions = model.predict(self.blocks, batch_size=TPU_BATCH_SIZE if CFG['TPU'] else GPU_BATCH_SIZE, verbose=self.verbose) # Example shape (self.blocks_counter+pad_value, 864, 3)
        predictions = tf.expand_dims(predictions, axis=-1) # Example shape (self.blocks_counter+pad_value, 864, 3, 1)
        predictions = tf.transpose(predictions, perm=[0, 1, 3, 2]) # Example shape (self.blocks_counter+pad_value, 864, 1, 3)
        predictions = tf.tile(predictions, multiples=[1, 1, CFG['patch_size'], 1]) # Example shape (self.blocks_counter+pad_value, 864, 18, 3)
        predictions = tf.reshape(predictions, shape=(predictions.shape[0], predictions.shape[1]*predictions.shape[2], 3)) # Example shape (self.blocks_counter+pad_value, 15552, 3)
        predictions = predictions.numpy()
        
        # get predictions
        def create_target(description):
            series, series_blocks = description['series'].copy(), description['series_blocks']
            
            values = np.zeros((series_blocks[-1][2], 4))
            for series_block in series_blocks:
                i, begin, end = series_block
                values[begin:end, 0:3] += predictions[i]
                values[begin:end, 3] += 1

            values = values[:len(series)]
            
            series['StartHesitation_prediction'] = values[:, 0] / values[:, 3]
            series['Turn_prediction'] = values[:, 1] / values[:, 3]
            series['Walking_prediction'] = values[:, 2] / values[:, 3]
            series['Prediction_count'] = values[:, 3]
            series['Event_prediction'] = series[['StartHesitation_prediction', 'Turn_prediction', 'Walking_prediction']].aggregate('max', axis=1)
            
            return series
            
        targets = Parallel(n_jobs=-1)(delayed(create_target)(self.descriptions[i]) for i in tqdm(range(len(self.descriptions)), disable=1-self.verbose))
        targets = pd.concat(targets)
        
        return targets

# load models
model = FOGModel()
model.build(input_shape=(GPU_BATCH_SIZE, CFG['block_size'] // CFG['patch_size'], CFG['patch_size']*3))
model.load_weights('/kaggle/input/my-fog-dataset/LSTM_1.h5') # TDCSFOG weights


# Prediction
w = 0.25
for Id in tsfog_ids:
    targets = PredictionFnCallback_tdcs_1(prediction_ids=[Id], model=model).prediction()
    submission = pd.DataFrame({'Id': (targets['Id'].values + '_' + targets['Time'].astype('str')).values,
                               'StartHesitation': targets['StartHesitation_prediction'].values*w,
                               'Turn': targets['Turn_prediction'].values*w,
                               'Walking': targets['Walking_prediction'].values*w,
                              })
    
    all_submissions.append(submission)

'''
============================================================================================================================================================
tdcsfog model_2
============================================================================================================================================================
'''
CFG = {'TPU': 0, 
       'block_size': 15552, 
       'block_stride': 15552//64,
       'patch_size': 18, 

       'fog_model_dim': 256,
       'fog_model_num_heads': 6,
       'fog_model_num_encoder_layers': 3,
       'fog_model_num_lstm_layers': 2,
       'fog_model_first_dropout': 0.1,
       'fog_model_encoder_dropout': 0.1,
       'fog_model_mha_dropout': 0.0,
      }

assert CFG['block_size'] % CFG['patch_size'] == 0
assert CFG['block_size'] % CFG['block_stride'] == 0

GPU_BATCH_SIZE = 16
TPU_BATCH_SIZE = GPU_BATCH_SIZE*8

class PredictionFnCallback_tdcs_2(tf.keras.callbacks.Callback):
    
    def __init__(self, prediction_ids, model=None, verbose=0):
        
        if not model is None: self.model = model
        self.verbose = verbose
         
        def init(Id, path):
            series = pd.read_csv(path).reset_index(drop=True)
            series['Id'] = Id
            series['AccV'] = sample_normalize(series['AccV'].values)
            series['AccML'] = sample_normalize(series['AccML'].values)
            series['AccAP'] = sample_normalize(series['AccAP'].values)
            
            series_blocks=[]
            for block in get_blocks(series, ['AccV', 'AccML', 'AccAP']): 
                values = tf.reshape(block['values'], shape=(CFG['block_size'] // CFG['patch_size'], CFG['patch_size'], 3)) 
                values = tf.reshape(values, shape=(CFG['block_size'] // CFG['patch_size'], CFG['patch_size']*3)) 
                values = tf.concat([values, 
                                    tf.constant(0.0, dtype=tf.float32, shape=(CFG['block_size'] // CFG['patch_size'], 1)),
                                   ], axis=-1) 
                values = tf.expand_dims(values, axis=0) 
                
                self.blocks.append(values)
                series_blocks.append((self.blocks_counter, block['begin'], block['end']))
                self.blocks_counter += 1
            
            description = {}
            description['series'] = series
            description['series_blocks'] = series_blocks
            self.descriptions.append(description)
            
        self.descriptions = [] 
        self.blocks = [] 
        self.blocks_counter=0 
        
        notype_ids = prediction_ids
        notype_paths = [f'/kaggle/input/tlvmc-parkinsons-freezing-gait-prediction/test/tdcsfog/{notype_id}.csv' for notype_id in notype_ids]
        for notype_id, notype_path in tqdm(zip(notype_ids, notype_paths), total=len(notype_ids), desc='PredictionFnCallback Initialization', disable=1-verbose): 
            init(notype_id, notype_path)
            
        self.blocks = tf.concat(self.blocks, axis=0) 
        
        self.blocks = tf.pad(self.blocks, 
                             paddings=[[0, math.ceil(self.blocks_counter / (TPU_BATCH_SIZE if CFG['TPU'] else GPU_BATCH_SIZE))*(TPU_BATCH_SIZE if CFG['TPU'] else GPU_BATCH_SIZE)-self.blocks_counter], 
                                                    [0, 0], 
                                                    [0, 0],
                                      ])
        
        print(f'[PredictionFnCallback Initialization] [Series] {len(self.descriptions)} [Blocks] {self.blocks_counter}')
    
    def prediction(self):
        predictions = model.predict(self.blocks, batch_size=TPU_BATCH_SIZE if CFG['TPU'] else GPU_BATCH_SIZE, verbose=self.verbose)
        predictions = tf.expand_dims(predictions, axis=-1) 
        predictions = tf.transpose(predictions, perm=[0, 1, 3, 2]) 
        predictions = tf.tile(predictions, multiples=[1, 1, CFG['patch_size'], 1])
        predictions = tf.reshape(predictions, shape=(predictions.shape[0], predictions.shape[1]*predictions.shape[2], 3))
        predictions = predictions.numpy() 
        
        def create_target(description):
            series, series_blocks = description['series'].copy(), description['series_blocks']
            
            values = np.zeros((series_blocks[-1][2], 4))
            for series_block in series_blocks:
                i, begin, end = series_block
                values[begin:end, 0:3] += predictions[i]
                values[begin:end, 3] += 1

            values = values[:len(series)]
            
            series['StartHesitation_prediction'] = values[:, 0] / values[:, 3]
            series['Turn_prediction'] = values[:, 1] / values[:, 3]
            series['Walking_prediction'] = values[:, 2] / values[:, 3]
            series['Prediction_count'] = values[:, 3]
            
            return series
            
        targets = Parallel(n_jobs=-1)(delayed(create_target)(self.descriptions[i]) for i in tqdm(range(len(self.descriptions)), disable=1-self.verbose))
        targets = pd.concat(targets)
        
        return targets
    
model = FOGModel()
model.build(input_shape=(GPU_BATCH_SIZE, CFG['block_size'] // CFG['patch_size'], CFG['patch_size']*3+1))
model.load_weights('/kaggle/input/my-fog-dataset/LSTM_2.h5')

w = 0.35
for Id in tsfog_ids:
    targets = PredictionFnCallback_tdcs_2(prediction_ids=[Id], model=model).prediction()
    submission = pd.DataFrame({'Id': (targets['Id'].values + '_' + targets['Time'].astype('str')).values,
                               'StartHesitation': targets['StartHesitation_prediction'].values*w,
                               'Turn': targets['Turn_prediction'].values*w,
                               'Walking': targets['Walking_prediction'].values*w,
                              })
    
    all_submissions.append(submission)

'''
============================================================================================================================================================
tdcsfog model_3
============================================================================================================================================================
'''
CFG = {'TPU': 0,
       'block_size': 15552, 
       'block_stride': 15552//32,
       'patch_size': 18, 
       
       'fog_model_dim': 320,
       'fog_model_num_heads': 6,
       'fog_model_num_encoder_layers': 5,
       'fog_model_num_lstm_layers': 2,
       'fog_model_first_dropout': 0.1,
       'fog_model_encoder_dropout': 0.1,
       'fog_model_mha_dropout': 0.0,
      }

assert CFG['block_size'] % CFG['patch_size'] == 0
assert CFG['block_size'] % CFG['block_stride'] == 0

GPU_BATCH_SIZE = 4
TPU_BATCH_SIZE = GPU_BATCH_SIZE*8

WEIGHTS = '/kaggle/input/my-fog-dataset/LSTM_3.h5' 
    
model = FOGModel()
model.build(input_shape=(GPU_BATCH_SIZE, CFG['block_size'] // CFG['patch_size'], CFG['patch_size']*3))
if len(WEIGHTS): model.load_weights(WEIGHTS)    
    
w = 0.2
for Id in tsfog_ids:
    targets = PredictionFnCallback_tdcs_1(prediction_ids=[Id], model=model).prediction()
    submission = pd.DataFrame({'Id': (targets['Id'].values + '_' + targets['Time'].astype('str')).values,
                               'StartHesitation': targets['StartHesitation_prediction'].values*w,
                               'Turn': targets['Turn_prediction'].values*w,
                               'Walking': targets['Walking_prediction'].values*0,
                              })
    
    all_submissions.append(submission)

'''
============================================================================================================================================================
tdcsfog 模型4
============================================================================================================================================================
'''
CFG = {'TPU': 0,
       'block_size': 15552, 
       'block_stride': 15552//32,
       'patch_size': 18, 
       
       'fog_model_dim': 320,
       'fog_model_num_heads': 6,
       'fog_model_num_encoder_layers': 5,
       'fog_model_num_lstm_layers': 2,
       'fog_model_first_dropout': 0.1,
       'fog_model_encoder_dropout': 0.1,
       'fog_model_mha_dropout': 0.0,
      }

assert CFG['block_size'] % CFG['patch_size'] == 0
assert CFG['block_size'] % CFG['block_stride'] == 0

GPU_BATCH_SIZE = 4
TPU_BATCH_SIZE = GPU_BATCH_SIZE*8
    
model = FOGModel()
model.build(input_shape=(GPU_BATCH_SIZE, CFG['block_size'] // CFG['patch_size'], CFG['patch_size']*3))
model.load_weights('/kaggle/input/my-fog-dataset/LSTM_4.h5')

w = 0.2
for Id in tsfog_ids:
    targets = PredictionFnCallback_tdcs_1(prediction_ids=[Id], model=model).prediction()
    submission = pd.DataFrame({'Id': (targets['Id'].values + '_' + targets['Time'].astype('str')).values,
                               'StartHesitation': targets['StartHesitation_prediction'].values*w,
                               'Turn': targets['Turn_prediction'].values*0,
                               'Walking': targets['Walking_prediction'].values*w,
                              })
    
    all_submissions.append(submission)


'''
============================================================================================================================================================
tdcsfog model_5
============================================================================================================================================================
'''
CFG = {'TPU': 0, 
       'block_size': 15552, 
       'block_stride': 15552//64,
       'patch_size': 18, 

       'fog_model_dim': 320,
       'fog_model_num_heads': 6,
       'fog_model_num_encoder_layers': 5,
       'fog_model_num_lstm_layers': 3,
       'fog_model_first_dropout': 0.2,
       'fog_model_encoder_dropout': 0.2,
       'fog_model_mha_dropout': 0.1,
      }

assert CFG['block_size'] % CFG['patch_size'] == 0
assert CFG['block_size'] % CFG['block_stride'] == 0

GPU_BATCH_SIZE = 16
TPU_BATCH_SIZE = GPU_BATCH_SIZE*8

model = FOGModel()
model.build(input_shape=(GPU_BATCH_SIZE, CFG['block_size'] // CFG['patch_size'], CFG['patch_size']*3))
model.load_weights('/kaggle/input/my-fog-dataset/LSTM_5.h5')

w = 0.2
for Id in tsfog_ids:
    targets = PredictionFnCallback_tdcs_1(prediction_ids=[Id], model=model).prediction()
    submission = pd.DataFrame({'Id': (targets['Id'].values + '_' + targets['Time'].astype('str')).values,
                               'StartHesitation': targets['StartHesitation_prediction'].values*0,
                               'Turn': targets['Turn_prediction'].values*w,
                               'Walking': targets['Walking_prediction'].values*w,
                              })
    
    all_submissions.append(submission)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# defog
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=CFG['fog_model_num_heads'], key_dim=CFG['fog_model_dim'], dropout=CFG['fog_model_mha_dropout'])
        
        self.add = tf.keras.layers.Add()
        
        self.layernorm = tf.keras.layers.LayerNormalization()
        
        self.seq = tf.keras.Sequential([tf.keras.layers.Dense(CFG['fog_model_dim'], activation='relu'),
                                        tf.keras.layers.Dropout(CFG['fog_model_encoder_dropout']), 
                                        tf.keras.layers.Dense(CFG['fog_model_dim']), 
                                        tf.keras.layers.Dropout(CFG['fog_model_encoder_dropout']),
                                       ])
    
    def call(self, x):
        attn_output = self.mha(query=x, key=x, value=x)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        x = self.add([x, self.seq(x)])
        x = self.layernorm(x)
        
        return x

class FOGEncoder(tf.keras.Model):
    def __init__(self):
        super().__init__()
        
        self.first_linear = tf.keras.layers.Dense(CFG['fog_model_dim'])
        
        self.add = tf.keras.layers.Add()
        
        self.first_dropout = tf.keras.layers.Dropout(CFG['fog_model_first_dropout'])
        
        self.enc_layers = [EncoderLayer() for _ in range(CFG['fog_model_num_encoder_layers'])]
        
        self.lstm_layers = [tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(CFG['fog_model_dim'], return_sequences=True)) for _ in range(CFG['fog_model_num_lstm_layers'])]
        
        self.sequence_len = CFG['block_size'] // CFG['patch_size']
        self.pos_encoding = tf.Variable(initial_value=tf.random.normal(shape=(1, self.sequence_len, CFG['fog_model_dim']), stddev=0.02), trainable=True)
        
    def call(self, x, training=None): # (GPU_BATCH_SIZE, CFG['block_size'] // CFG['patch_size'], CFG['patch_size']*3), Example shape (4, 864, 42)
        x = x / 50.0
        x = self.first_linear(x) # (GPU_BATCH_SIZE, CFG['block_size'] // CFG['patch_size'], CFG['fog_model_dim']), Example shape (4, 864, 320)
          
        if training:
            random_pos_encoding = tf.roll(tf.tile(self.pos_encoding, multiples=[GPU_BATCH_SIZE, 1, 1]), 
                                          shift=tf.random.uniform(shape=(GPU_BATCH_SIZE,), minval=-self.sequence_len, maxval=0, dtype=tf.int32),
                                          axis=GPU_BATCH_SIZE * [1],
                                          )
            x = self.add([x, random_pos_encoding])
        
        else: 
            x = self.add([x, tf.tile(self.pos_encoding, multiples=[GPU_BATCH_SIZE, 1, 1])])
            
        x = self.first_dropout(x)
        
        for i in range(CFG['fog_model_num_encoder_layers']): x = self.enc_layers[i](x) # (GPU_BATCH_SIZE, CFG['block_size'] // CFG['patch_size'], CFG['fog_model_dim']), Example shape (4, 864, 320)
        for i in range(CFG['fog_model_num_lstm_layers']): x = self.lstm_layers[i](x) # (GPU_BATCH_SIZE, CFG['block_size'] // CFG['patch_size'], CFG['fog_model_dim']*2), Example shape (4, 864, 640)
            
        return x

class FOGModel_dense_4(tf.keras.Model):
    def __init__(self):
        super().__init__()
        
        self.encoder = FOGEncoder()
        # 3 labels with event label
        self.last_linear = tf.keras.layers.Dense(4) 
        
    def call(self, x): # (GPU_BATCH_SIZE, CFG['block_size'] // CFG['patch_size'], CFG['patch_size']*3), Example shape (4, 864, 42)
        x = self.encoder(x) # (GPU_BATCH_SIZE, CFG['block_size'] // CFG['patch_size'], CFG['fog_model_dim']*2), Example shape (4, 864, 640)
        x = self.last_linear(x) # (GPU_BATCH_SIZE, CFG['block_size'] // CFG['patch_size'], 3), Example shape (4, 864, 4)
        x = tf.nn.sigmoid(x)
        
        return x

'''
============================================================================================================================================================
defog model_6
============================================================================================================================================================
''' 
CFG = {'TPU': 0,
       'block_size': 12096, 
       'block_stride': 12096//48,
       'patch_size': 14, 
       
       'fog_model_dim': 320,
       'fog_model_num_heads': 6,
       'fog_model_num_encoder_layers': 5,
       'fog_model_num_lstm_layers': 2,
       'fog_model_first_dropout': 0.1,
       'fog_model_encoder_dropout': 0.1,
       'fog_model_mha_dropout': 0.0,
      }

assert CFG['block_size'] % CFG['patch_size'] == 0
assert CFG['block_size'] % CFG['block_stride'] == 0

GPU_BATCH_SIZE = 4
TPU_BATCH_SIZE = GPU_BATCH_SIZE*8


class PredictionFnCallback_de_6(tf.keras.callbacks.Callback):
    
    def __init__(self, prediction_ids, model=None, verbose=0):
        
        if not model is None: self.model = model
        self.verbose = verbose
         
        def init(Id, path):
            series = pd.read_csv(path).reset_index(drop=True)
            series['Id'] = Id
            series['AccV'] = sample_normalize(series['AccV'].values)
            series['AccML'] = sample_normalize(series['AccML'].values)
            series['AccAP'] = sample_normalize(series['AccAP'].values)
            
            series_blocks=[]
            for block in get_blocks(series, ['AccV', 'AccML', 'AccAP']): # Example shape (12096, 3)
                values = tf.reshape(block['values'], shape=(CFG['block_size'] // CFG['patch_size'], CFG['patch_size'], 3)) # Example shape (864, 14, 3)
                values = tf.reshape(values, shape=(CFG['block_size'] // CFG['patch_size'], CFG['patch_size']*3)) # Example shape (864, 42)
                values = tf.expand_dims(values, axis=0) # Example shape (1, 864, 42)
                
                self.blocks.append(values)
                series_blocks.append((self.blocks_counter, block['begin'], block['end']))
                self.blocks_counter += 1
            
            description = {}
            description['series'] = series
            description['series_blocks'] = series_blocks
            self.descriptions.append(description)
            
        self.descriptions = [] 
        self.blocks = []
        self.blocks_counter=0
                
        defog_ids = prediction_ids
        defog_paths = [f'/kaggle/input/tlvmc-parkinsons-freezing-gait-prediction/test/defog/{defog_id}.csv' for defog_id in defog_ids]
        for defog_id, defog_path in tqdm(zip(defog_ids, defog_paths), total=len(defog_ids), desc='PredictionFnCallback Initialization', disable=1-verbose): 
            init(defog_id, defog_path)
                
        self.blocks = tf.concat(self.blocks, axis=0)  # Example shape (self.blocks_counter, 864, 42)
        
        self.blocks = tf.pad(self.blocks, 
                             paddings=[[0, math.ceil(self.blocks_counter / (TPU_BATCH_SIZE if CFG['TPU'] else GPU_BATCH_SIZE))*(TPU_BATCH_SIZE if CFG['TPU'] else GPU_BATCH_SIZE)-self.blocks_counter], 
                                                    [0, 0], 
                                                    [0, 0],
                                      ]) # Example shape (self.blocks_counter+pad_value, 864, 42)
        
        print(f'\n[PredictionFnCallback Initialization] [Series] {len(self.descriptions)} [Blocks] {self.blocks_counter}\n')
    
    def prediction(self):
        predictions = model.predict(self.blocks, batch_size=TPU_BATCH_SIZE if CFG['TPU'] else GPU_BATCH_SIZE, verbose=self.verbose) # Example shape (self.blocks_counter+pad_value, 864, 4)
        predictions = predictions[:, :, :3] # Example shape (self.blocks_counter+pad_value, 864, 3)
        predictions = tf.expand_dims(predictions, axis=-1) # Example shape (self.blocks_counter+pad_value, 864, 3, 1)
        predictions = tf.transpose(predictions, perm=[0, 1, 3, 2]) # Example shape (self.blocks_counter+pad_value, 864, 1, 3)
        predictions = tf.tile(predictions, multiples=[1, 1, CFG['patch_size'], 1]) # Example shape (self.blocks_counter+pad_value, 864, 14, 3)
        predictions = tf.reshape(predictions, shape=(predictions.shape[0], predictions.shape[1]*predictions.shape[2], 3)) # Example shape (self.blocks_counter+pad_value, 12096, 3)
        predictions = predictions.numpy()

        def create_target(description):
            series, series_blocks = description['series'].copy(), description['series_blocks']
            
            values = np.zeros((series_blocks[-1][2], 4))
            for series_block in series_blocks:
                i, begin, end = series_block
                values[begin:end, 0:3] += predictions[i]
                values[begin:end, 3] += 1

            values = values[:len(series)]
            
            series['StartHesitation_prediction'] = values[:, 0] / values[:, 3]
            series['Turn_prediction'] = values[:, 1] / values[:, 3]
            series['Walking_prediction'] = values[:, 2] / values[:, 3]
            series['Prediction_count'] = values[:, 3]
            
            return series
            
        targets = Parallel(n_jobs=-1)(delayed(create_target)(self.descriptions[i]) for i in tqdm(range(len(self.descriptions)), disable=1-self.verbose))
        targets = pd.concat(targets).reset_index(drop=True)
        
        return targets


WEIGHTS = '/kaggle/input/my-fog-dataset/LSTM_6.h5'

model = FOGModel_dense_4()
model.build(input_shape=(GPU_BATCH_SIZE, CFG['block_size'] // CFG['patch_size'], CFG['patch_size']*3))
if len(WEIGHTS): model.load_weights(WEIGHTS)

w = 0.3
for Id in defog_ids:
    targets = PredictionFnCallback_de_6(prediction_ids=[Id], model=model).prediction()
    submission = pd.DataFrame({'Id': (targets['Id'].values + '_' + targets['Time'].astype('str')).values,
                               'StartHesitation': targets['StartHesitation_prediction'].values*w,
                               'Turn': targets['Turn_prediction'].values*w,
                               'Walking': targets['Walking_prediction'].values*w,
                              })
    
    all_submissions.append(submission)

'''
============================================================================================================================================================
defog model_7
============================================================================================================================================================
'''
SUB_PATH = "/kaggle/input/tlvmc-parkinsons-freezing-gait-prediction/sample_submission.csv"
DEFOG_DATA_PATH = "/kaggle/input/tlvmc-parkinsons-freezing-gait-prediction/test/defog/*.csv"

# If your GPU has enough memory, you can use 'cuda'
device = torch.device('cpu')
bs = 32

defog_path8 = [f"/kaggle/input/my-fog-dataset/{i}.pth" for i in range(5)]

def preprocess(numerical_array, 
               mask_array,
               ):
    
    attention_mask = mask_array == 0

    return {
        'input_data_numerical_array': numerical_array,
        'input_data_mask_array': mask_array,
        'attention_mask': attention_mask,
    }

class FogDataset(Dataset):
    def __init__(self, numerical_array, 
                 mask_array,
                 train = True, y = None):
        self.numerical_array = numerical_array
        self.mask_array = mask_array
        self.train = train
        self.y = y
    
    def __len__(self):
        return len(self.numerical_array)

    def __getitem__(self, item):
        data = preprocess(
            self.numerical_array[item],
            self.mask_array[item],
            
        )

        # Return the processed data where the lists are converted to `torch.tensor`s
        if self.train : 
            return {
              'input_data_numerical_array': torch.tensor(data['input_data_numerical_array'],dtype=torch.float32),
              'input_data_mask_array':torch.tensor(data['input_data_mask_array'], dtype=torch.long),  
              'attention_mask': torch.tensor(data["attention_mask"], dtype=torch.bool),
              "y":torch.tensor(self.y[item], dtype=torch.float32)
               }
        else:
            return {
             'input_data_numerical_array': torch.tensor(data['input_data_numerical_array'],dtype=torch.float32),
              'input_data_mask_array':torch.tensor(data['input_data_mask_array'], dtype=torch.long),  
              'attention_mask': torch.tensor(data["attention_mask"], dtype=torch.bool),
               }
    
class DefogRnnModel(nn.Module):
    def __init__(
        self, dropout=0.2,
        input_numerical_size=9,
        numeraical_linear_size = 64,
        model_size = 128,
        linear_out = 128,
        out_size=3):
        super(DefogRnnModel, self).__init__()
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

def make_pred(test_loader,model):
    test_preds = []
    tk0 = tqdm(test_loader, total=len(test_loader))
    with torch.no_grad():  # Do not calculate gradient since we are only predicting
        # Predicting on validation set
        for d in tk0:
            input_data_numerical_array = d['input_data_numerical_array'].to(device)
            input_data_mask_array = d['input_data_mask_array'].to(device)
            attention_mask = d['attention_mask'].to(device)
            output = model(input_data_numerical_array, 
                       input_data_mask_array,
                       attention_mask)
            test_preds.append(output.sigmoid().cpu().numpy())
    test_preds = np.concatenate(test_preds,axis=0)
    return test_preds

sub = pd.read_csv(SUB_PATH)

defog_model_list7 = []
for i in defog_path8:
    model = DefogRnnModel()
    model.load_state_dict(torch.load(i))
    model = model.to(device)
    model.eval()
    defog_model_list7.append(model)

th_len = 200000
w = 0.7

defog_list = glob.glob(DEFOG_DATA_PATH)
cols = ["AccV","AccML","AccAP"]
num_cols = ["AccV","AccML","AccAP",'AccV_lag_diff',
            'AccV_lead_diff', 'AccML_lag_diff', 'AccML_lead_diff',
            'AccAP_lag_diff', 'AccAP_lead_diff']
for p in tqdm(defog_list):
    id_values = p.split("/")[-1].split(".")[0]
    df = pd.read_csv(p)
    if len(df) > th_len:
        seq_len = 30000
        shift = 15000
        offset = 7500
    else:
        seq_len = 15000
        shift = 7500
        offset = 3750
    batch = (len(df)-1) // shift
    if batch == 0:
        batch = 1
    for c in cols:
        df[f"{c}_lag_diff"] = df[c].diff()
        df[f"{c}_lead_diff"] = df[c].diff(-1)
    sc = StandardScaler() # equal to sample_normalize function
    df[num_cols] = sc.fit_transform(df[num_cols].values)
    df[num_cols] = df[num_cols].fillna(0)
    num = df[num_cols].values
    time = df["Time"].values
    
    num_array = np.zeros([batch,seq_len,9])
    mask_array = np.zeros([batch,seq_len],dtype=int)
    time_array = np.zeros([batch,seq_len],dtype=int)
    pred_use_array = np.zeros([batch,seq_len],dtype=int)
    
    if len(df) <= seq_len:
        b = 0
        num_len = len(num)
        num_array[b,:num_len,:] = num
        time_array[b,:num_len] = time
        mask_array[b,:num_len] = 1
        pred_use_array[b,:num_len] = 1
    else:
        for n,b in enumerate(range(batch)):
            if b == (batch - 1):
                num_ = num[b*shift : ]
                time_ = time[b*shift : ]
                num_len = len(num_)

                num_array[b,:num_len,:] = num_
                time_array[b,:num_len] = time_
                mask_array[b,:num_len] = 1
                pred_use_array[b,offset:num_len] = 1
            elif b == 0:
                num_ = num[b*shift:b*shift+seq_len]
                time_ = time[b*shift:b*shift + seq_len]

                num_array[b,:,:] = num_
                time_array[b,:] = time_
                mask_array[b,:] = 1
                pred_use_array[b,:shift+offset] = 1
            else:
                num_ = num[b*shift:b*shift+seq_len]
                time_ = time[b*shift:b*shift + seq_len]

                num_array[b,:,:] = num_
                time_array[b,:] = time_
                mask_array[b,:] = 1
                pred_use_array[b,offset:shift+offset] = 1  
    
    test_ = FogDataset(num_array,
                       mask_array,
                       train=False)
    test_loader = DataLoader(dataset=test_, 
                        batch_size=bs, 
                        shuffle = False)
    for n,m in enumerate(defog_model_list7):
        if n == 0:
            pred = make_pred(test_loader,m) / len(defog_model_list7)
        else:
            pred += make_pred(test_loader,m) / len(defog_model_list7)
    pred_list = []
    for i in range(batch):
        mask_ = pred_use_array[i]
        pred_ = pred[i,mask_ == 1,:]
        time_ = time_array[i, mask_ == 1]
        df_ = pd.DataFrame()
        df_["StartHesitation"] = pred_[:,0] * w
        df_["Turn"] = pred_[:,1] * w
        df_["Walking"] = pred_[:,2] * w
        df_["Time"] = time_
        df_["Id"] = id_values
        df_["Id"] = df_["Id"].astype(str) + "_" + df_["Time"].astype(str)
        pred_list.append(df_)
    pred = pd.concat(pred_list).reset_index(drop=True)

    submission = pd.DataFrame({'Id': pred['Id'],
                               'StartHesitation': pred['StartHesitation'].values,
                               'Turn': pred['Turn'].values,
                               'Walking': pred['Walking'].values,
                              })
    
    all_submissions.append(submission)

'''
============================================================================================================================================================
submission
============================================================================================================================================================
''' 
submission = pd.concat(all_submissions).reset_index(drop=True)
submission = submission.groupby('Id')[['StartHesitation', 'Turn', 'Walking']].sum().reset_index()
submission[['Id', 'StartHesitation', 'Turn', 'Walking']].to_csv('submission.csv', index=False)
