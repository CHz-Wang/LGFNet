import os 
import math
import numpy as np 
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
import random
import warnings
from scipy import signal

CFG = {'TPU': 0, 
       'block_size': 15552, 
       'block_stride': 15552//16,
       'patch_size': 18, 

       'fog_model_dim': 256,
       'fog_model_num_heads': 6,
       'fog_model_num_encoder_layers': 4,
       'fog_model_num_lstm_layers': 3,
       'fog_model_first_dropout': 0.1,
       'fog_model_encoder_dropout': 0.1,
       'fog_model_mha_dropout': 0.0,
      }

assert CFG['block_size'] % CFG['patch_size'] == 0
assert CFG['block_size'] % CFG['block_stride'] == 0

'''
Mean-std normalization function. 
Example input: shape (5000), dtype np.float32
Example output: shape (5000), dtype np.float32

Used to normalize AccV, AccML, AccAP values.

'''

def sample_normalize(sample):
    mean = tf.math.reduce_mean(sample)
    std = tf.math.reduce_std(sample)
    sample = tf.math.divide_no_nan(sample-mean, std)
    
    return sample.numpy()

'''
Function for splitting a series into blocks. Blocks can overlap. 
How the function works:
Suppose we have a series with AccV, AccML, AccAP columns and len of 50000, that is (50000, 3). 
First, the series is padded so that the final length is divisible by CFG['block_size'] = 15552. Now the series shape is (62208, 3).
Then we get blocks: first block is series[0:15552, :], second block is series[972:16524, :], ... , last block is series[46656:62208, :].

'''

def get_blocks(series, columns):
    series = series.copy()
    series = series[columns]
    series = series.values
    series = series.astype(np.float32)
    
    block_count = math.ceil(len(series) / CFG['block_size'])
    
    series = np.pad(series, pad_width=[[0, block_count*CFG['block_size']-len(series)], [0, 0]])
    
    block_begins = list(range(0, len(series), CFG['block_stride']))
    block_begins = [x for x in block_begins if x+CFG['block_size'] <= len(series)]
    
    blocks = []
    for begin in block_begins:
        values = series[begin:begin+CFG['block_size']]
        blocks.append({'begin': begin,
                       'end': begin+CFG['block_size'],
                       'values': values})
    
    return blocks

'''
Train and inference batch size

'''

GPU_BATCH_SIZE = 16
TPU_BATCH_SIZE = GPU_BATCH_SIZE*8


if CFG['TPU']:
    !pip install -q /lib/wheels/tensorflow-2.9.1-cp38-cp38-linux_x86_64.whl
    !pip install -qU scikit-learn
    
import numpy as np 
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy

from tqdm import tqdm
from itertools import cycle
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score

if CFG['TPU']:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect(tpu='local') 
    tpu_strategy = tf.distribute.TPUStrategy(tpu)

warnings.filterwarnings("ignore")
pd.set_option('display.max_colwidth', None)

def folder(path): 
    if not os.path.exists(path): os.makedirs(path)
        
def plot(e, size=(20, 4)):
    plt.figure(figsize=size)
    plt.plot(e)
    plt.show()


################################################################################################################################################################ 模型
'''
The transformer encoder layer
For more details, see https://arxiv.org/pdf/1706.03762.pdf [Attention Is All You Need]

'''

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=CFG['fog_model_num_heads'], key_dim=CFG['fog_model_dim'], dropout=CFG['fog_model_mha_dropout'])
        
        self.add = tf.keras.layers.Add()
        
        self.layernorm = tf.keras.layers.LayerNormalization()
        
        self.seq = tf.keras.Sequential([tf.keras.layers.Dense(CFG['fog_model_dim'], activation='relu'), 
                                        tf.keras.layers.Dropout(CFG['fog_model_encoder_dropout']), 
                                        tf.keras.layers.Dense(CFG['fog_model_dim'], activation='relu'), 
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
    
'''
FOGEncoder is a combination of transformer encoder (D=320, H=6, L=5) and two BidirectionalLSTM layers

'''

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
        
    def call(self, x, training=None): # (GPU_BATCH_SIZE, CFG['block_size'] // CFG['patch_size'], CFG['patch_size']*3), Example shape (4, 864, 54)
        x = x / 25.0 # Normalization attempt in the segment [-1, 1]
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
        
        for i in range(CFG['fog_model_num_encoder_layers']): x = self.enc_layers[i](x) # (GPU_BATCH_SIZE, CFG['block_size'] // CFG['patch_size'], CFG['fog_model_dim']), Example shape (4, 864, 320)
        for i in range(CFG['fog_model_num_lstm_layers']): x = self.lstm_layers[i](x) # (GPU_BATCH_SIZE, CFG['block_size'] // CFG['patch_size'], CFG['fog_model_dim']*2), Example shape (4, 864, 640)
            
        return x
    
class FOGModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        
        self.encoder = FOGEncoder()
        self.least_linear = tf.keras.layers.Dense(100)
        self.last_linear = tf.keras.layers.Dense(3) 
        
    def call(self, x): # (GPU_BATCH_SIZE, CFG['block_size'] // CFG['patch_size'], CFG['patch_size']*3), Example shape (4, 864, 54)
        x = self.encoder(x) # (GPU_BATCH_SIZE, CFG['block_size'] // CFG['patch_size'], CFG['fog_model_dim']*2), Example shape (4, 864, 640)
        x = self.least_linear(x)
        x = self.last_linear(x) # (GPU_BATCH_SIZE, CFG['block_size'] // CFG['patch_size'], 3), Example shape (4, 864, 3)
        x = tf.nn.sigmoid(x) # Sigmoid activation
        
        return x


################################################################################################################################################################ 数据预处理
'''
Create train blocks with AccV, AccML, AccAP, StartHesitation, Turn, Walking, Valid, Mask columns and save in the directory

'''

save_path = '/kaggle/working/train/tdcsfog'; folder(save_path); 
tdcsfog_metadata = pd.read_csv('/kaggle/input/tlvmc-parkinsons-freezing-gait-prediction/tdcsfog_metadata.csv').set_index('Id')

blocks_descriptions = []
for Id in tqdm(tdcsfog_metadata.index, total=len(tdcsfog_metadata.index), desc='Preparing'):
    series = pd.read_csv(f'/kaggle/input/tlvmc-parkinsons-freezing-gait-prediction/train/tdcsfog/{Id}.csv')

    '''
    # label shift
    series["Event"]=series[["StartHesitation","Turn","Walking"]].max(axis=1)
    arr=series[["Event","Valid"]].astype(int).values
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
    series[["StartHesitation","Turn","Walking"]]=series[["StartHesitation","Turn","Walking"]].shift(-ind).fillna(0)'''

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
    
    series['AccV'] = sample_normalize(series['AccV'].values)
    series['AccML'] = sample_normalize(series['AccML'].values)
    series['AccAP'] = sample_normalize(series['AccAP'].values)
    series['Valid'] = 1
    series['Mask'] = 1

    blocks = get_blocks(series, ['AccV', 'AccML', 'AccAP', 'StartHesitation', 'Turn', 'Walking', 'Valid', 'Mask'])

    for block_count, block in enumerate(blocks):
        fname, values = f'{Id}_{block_count}.npy', block['values']
        block_description = {}
        block_description['Id'] = Id
        block_description['Count'] = block_count
        block_description['File'] = fname
        block_description['Path'] = f'{save_path}/{fname}'
        block_description['Source'] = 'tsfog'
        block_description['StartHesitation_size'] = np.sum(values[:, 3])
        block_description['Turn_size'] = np.sum(values[:, 4])
        block_description['Walking_size'] = np.sum(values[:, 5])
        block_description['Valid_size'] = np.sum(values[:, 6])
        block_description['Mask_size'] = np.sum(values[:, 7])

        blocks_descriptions.append(block_description)
        np.save(f'{save_path}/{fname}', values)

blocks_descriptions = pd.DataFrame(blocks_descriptions)


################################################################################################################################################################ 训练数据
'''
Selecting validation subjects
FOGModel train data preparing

'''

def write_to_ram(fog):
    fog = fog[['Id', 'Count', 'Path']]
    
    for _, row in tqdm(fog.iterrows(), total=len(fog), desc='Write'):
        Id, Count, path = row['Id'], row['Count'], row['Path']
        
        # Read data
        series = np.load(path) # ['AccV', 'AccML', 'AccAP', 'StartHesitation', 'Turn', 'Walking', 'Valid', 'Mask']

        # Create patches
        series = tf.reshape(series, shape=(CFG['block_size'] // CFG['patch_size'], CFG['patch_size'], series.shape[1]))

        # Create input
        series_input = series[:, :, 0:3]
        series_input = tf.reshape(series_input, shape=(CFG['block_size'] // CFG['patch_size'], -1))

        # Create target
        series_target = series[:, :, 3:]
        series_target = tf.transpose(series_target, perm=[0, 2, 1])
        series_target = tf.reduce_max(series_target, axis=-1)
        series_target = tf.cast(series_target, tf.int64)

        RAM[(Id, Count)] = (series_input, series_target)
        
val_subjects = ['07285e', '220a17', '54ee6e', '312788', '24a59d', '4bb5d0', '48fd62', '79011a', '7688c1']

train_ids = tdcsfog_metadata[tdcsfog_metadata['Subject'].apply(lambda x: x not in val_subjects)].index.tolist()
val_ids = tdcsfog_metadata[tdcsfog_metadata['Subject'].apply(lambda x: x in val_subjects)].index.tolist()

train_blocks_descriptions = blocks_descriptions[blocks_descriptions['Id'].apply(lambda x: x in train_ids)]

RAM = {} 
write_to_ram(train_blocks_descriptions)

print(f'\n[Train ids] {len(train_ids)} [Val ids] {len(val_ids)} ({100*len(val_ids)/(len(train_ids)+len(val_ids)):.1f})')
print(f'[Train blocks] {len(train_blocks_descriptions )}\n')


'''
Create a random train dataset from train_blocks_descriptions DataFrame

'''

def read(row):
    
    def read_from_ram(Id, Count):  
        series_inputs, series_targets = RAM[(Id.numpy().decode('utf-8'), Count.numpy())]
        series_targets = series_targets.numpy().astype(np.float32)
        
        return series_inputs, series_targets

    [series_input, series_target] = tf.py_function(read_from_ram, [row['Id'], row['Count']], [tf.float32, tf.float32])
    series_input.set_shape(shape=(CFG['block_size'] // CFG['patch_size'], CFG['patch_size']*3))
    series_target.set_shape(shape=(CFG['block_size'] // CFG['patch_size'], 5))
    
    return series_input, series_target

groups = [group.aggregate(dict, axis=1).tolist() for Id, group in train_blocks_descriptions.groupby('Id')]
random.shuffle(groups)
groups = cycle(groups)

dataset, iterator = [], 0
while len(dataset) <= 500000:
    group = next(groups)
    sample = random.choice(group)
    dataset.append(sample)
    iterator += 1
    
dataset = tf.data.Dataset.from_tensor_slices(dict(pd.DataFrame(dataset)))
dataset = dataset.map(read).batch(TPU_BATCH_SIZE if CFG['TPU'] else GPU_BATCH_SIZE, drop_remainder=True)


################################################################################################################################################################ 训练功能
'''
loss_function args exp

real is a tensor with the shape (GPU_BATCH_SIZE, CFG['block_size'] // CFG['patch_size'], 5) where the last axis means:
0 - StartHesitation 
1 - Turn
2 - Walking
3 - Valid
4 - Mask

output is a tensor with the shape (GPU_BATCH_SIZE, CFG['block_size'] // CFG['patch_size'], 3) where the last axis means:
0 - StartHesitation predicted
1 - Turn predicted
2 - Walking predicted

'''

ce = tf.keras.losses.BinaryCrossentropy(reduction='none')

def focal_loss(pred, y, alpha=0.25, gamma=2):
    zeros = tf.zeros_like(pred, dtype=pred.dtype)
    pos_p_sub = tf.where(y > zeros, y - pred, zeros) # positive sample 寻找正样本，并进行填充
    neg_p_sub = tf.where(y > zeros, zeros, pred) # negative sample 寻找负样本，并进行填充
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.math.log1p(tf.clip_by_value(pred, 1e-8, 1.0)) \
                              - (1 - alpha) * (neg_p_sub ** gamma) * tf.math.log1p(tf.clip_by_value(1.0 - pred, 1e-8, 1.0))

    return tf.reduce_mean(per_entry_cross_ent)

def loss_function(real, output, name='loss_function_2'):
    loss = ce(tf.expand_dims(real[:, :, 0:3], axis=-1), tf.expand_dims(output, axis=-1)) # Example shape (32, 864, 3)
    #mse = tf.keras.losses.mean_squared_error(tf.expand_dims(real[:, :, 0:3], axis=-1), tf.expand_dims(output, axis=-1))
    fl = focal_loss(tf.expand_dims(real[:, :, 0:3], axis=-1), tf.expand_dims(output, axis=-1))

    mask = tf.math.multiply(real[:, :, 3], real[:, :, 4]) # Example shape (32, 864)
    mask = tf.cast(mask, dtype=loss.dtype)
    mask = tf.expand_dims(mask, axis=-1) # Example shape (32, 864, 1)
    mask = tf.tile(mask, multiples=[1, 1, 3]) # Example shape (32, 864, 3)
    loss *= mask # Example shape (32, 864, 3)
    fl *= mask

    return (tf.reduce_sum(loss)*0.8 + tf.reduce_sum(fl)*0.2) / tf.reduce_sum(mask)

'''
Simple learning rate schedule with warm up steps

'''
        
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, warmup_steps=1):
        super(CustomSchedule, self).__init__()

        self.initial_lr = tf.cast(initial_lr, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        return tf.math.minimum(self.initial_lr, self.initial_lr * (step/self.warmup_steps))  
    

'''
PredictionFnCallback is used for:
1. Loading validation data
2. FOGModel data preparation
3. Prediction
4. Scoring and save

'''

class PredictionFnCallback(tf.keras.callbacks.Callback):
    
    def __init__(self, model=None, verbose=0):
        
        if not model is None: self.model = model
        self.verbose = verbose
         
        def init(Id, path):
            series = pd.read_csv(path).reset_index(drop=True)
            series['Id'] = Id

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

            series['AccV'] = sample_normalize(series['AccV'].values)
            series['AccML'] = sample_normalize(series['AccML'].values)
            series['AccAP'] = sample_normalize(series['AccAP'].values)
            series['Event'] = series[['StartHesitation', 'Turn', 'Walking']].aggregate('max', axis=1)
            
            series_blocks=[]
            for block in get_blocks(series, ['AccV', 'AccML', 'AccAP']): # Example shape (15552, 3)
                values = tf.reshape(block['values'], shape=(CFG['block_size'] // CFG['patch_size'], CFG['patch_size'], 3)) # Example shape (864, 18, 3)
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
        self.blocks = [] # Validation data blocks
        self.blocks_counter=0 # Blocks counter
        
        tsfog_ids = val_ids
        tsfog_paths = [f'/kaggle/input/tlvmc-parkinsons-freezing-gait-prediction/train/tdcsfog/{tsfog_id}.csv' for tsfog_id in tsfog_ids]
        for tsfog_id, tsfog_path in tqdm(zip(tsfog_ids, tsfog_paths), total=len(tsfog_ids), desc='PredictionFnCallback Initialization', disable=1-verbose): 
            init(tsfog_id, tsfog_path)
            
        self.blocks = tf.concat(self.blocks, axis=0) # Example shape (self.blocks_counter, 864, 54)
        
        '''
        self.blocks is padded so that the final length is divisible by inference batch size for error-free operation of model.predict function
        Padded values have no effect on the predictions
        
        '''
        
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
        
        '''
        The following function aggregates predictions blocks and creates dataframes with StartHesitation_prediction, Turn_prediction, Walking_prediction columns.
        
        '''
        
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
    
    def on_epoch_end(self, epoch, logs=None):
        scores=[]
        scores.append(f'{(epoch+1):03d}')
        
        loss = logs['loss'] if epoch >= 0 else 1.0
        
        targets = self.prediction()
        
        # Score
            
        StartHesitation_mAP = average_precision_score(targets['StartHesitation'], targets['StartHesitation_prediction'])
        Turn_mAP = average_precision_score(targets['Turn'], targets['Turn_prediction'])
        Walking_mAP = average_precision_score(targets['Walking'], targets['Walking_prediction'])
        mAP = (Walking_mAP+Turn_mAP+StartHesitation_mAP)/3

        print(f'\n\n[0] StartHesitation mAP - {StartHesitation_mAP:.3f} Turn mAP - {Turn_mAP:.3f} Walking mAP - {Walking_mAP:.3f} mAP - {mAP:.3f}')
        
        scores.append(f'{mAP:.3f}')
        
        # Score
        
        Event_mAP = average_precision_score(targets['Event'], targets['Event_prediction'])
        
        print(f'[1] Event mAP - {Event_mAP:.3f}\n')
        
        scores.append(f'{Event_mAP:.3f}')
        
        # Save
        
        scores.append(f'{loss:.4f}')
        
        save_name = '_'.join(scores)
        save_path = f'/kaggle/working/{save_name}_model.h5'
        self.model.save_weights(save_path)


################################################################################################################################################################ 训练
LEARNING_RATE = 0.01/128
STEPS_PER_EPOCH = 96
WARMUP_STEPS = 64
EPOCHS = 300
WEIGHTS = ''

if CFG['TPU']:
    with tpu_strategy.scope():
        model = FOGModel()
        model.build(input_shape=(GPU_BATCH_SIZE, CFG['block_size'] // CFG['patch_size'], CFG['patch_size']*3))
        if len(WEIGHTS): model.load_weights(WEIGHTS)
        model.compile(loss=loss_function, optimizer=tf.keras.optimizers.Adam(learning_rate=CustomSchedule(LEARNING_RATE, WARMUP_STEPS), beta_1=0.9, beta_2=0.98, epsilon=1e-9))
        !rm -r /kaggle/working/*
        model.fit(dataset, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, callbacks=[PredictionFnCallback()])
else:
    model = FOGModel()
    model.build(input_shape=(GPU_BATCH_SIZE, CFG['block_size'] // CFG['patch_size'], CFG['patch_size']*3))
    if len(WEIGHTS): model.load_weights(WEIGHTS)
    model.compile(loss=loss_function, optimizer=tf.keras.optimizers.Adam(learning_rate=CustomSchedule(LEARNING_RATE, WARMUP_STEPS), beta_1=0.9, beta_2=0.98, epsilon=1e-8))
    !rm -r /kaggle/working/*
    model.fit(dataset, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, callbacks=[PredictionFnCallback()])


################################################################################################################################################################ 模型保存
'''
Search for saved models in the working directory and sort them

'''

models = []
for fname in os.listdir('/kaggle/working/'):
    if 'model.h5' in fname:
        m = {}
        m['Path'] = '/kaggle/working/' + fname
        for i, elem in enumerate(fname.split('_')): 
            try:
                m[i+1] = float(elem)
            except:
                m[i+1] = elem
        models.append(m)

if len(models): 
    models = pd.DataFrame(models)
    plot(models.sort_values(1)[2].values)
    models = models.sort_values(2, ascending=False)
    display(models.head(15))