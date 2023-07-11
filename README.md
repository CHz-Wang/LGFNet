# Dataset
Data can be obtained from the Kaggle competition [here](https://www.kaggle.com/competitions/tlvmc-parkinsons-freezing-gait-prediction/data).<br/>
You can also reach the competition case homepage through the above link.
# Easy Run
If you want to simply replicate LGFNet, simply run `submit code.py` in Kaggle notebook.<br/>
Models you need are available in the `models` folder. 
# Train your own models
If you want to train the model yourself, you can run `train_GRUbased.py` and `train_LSTMbased.py`.<br/><br/>
**ATTENTION!**<br/><br/>
`train_GRUbased.py` only train on `Defog` dataset on the 1st stage.<br/>
In order to train on the 2nd stage, you should first get the pseudo label of `Notype` dataset.<br/>
Then `concat` with `Defog`.<br/>
The training code of two stages is the same.<br/><br/>
# Contact Info
If you have any question, please contact as at `1222046010@njupt.edu.cn`
