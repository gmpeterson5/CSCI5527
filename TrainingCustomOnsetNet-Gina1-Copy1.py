#!/usr/bin/env python
# coding: utf-8

# In[1]:


import getpass
import os

import logging
## Set default log level to WARNING, set this to INFO if you want more messages
logging.basicConfig(level=logging.WARNING)

from onset_train import train_onset_model, parse_args
from onset_net_gina5 import OnsetNet_Gina5

SM_DIR='/users/6/croal008/5527/ddc'
DATASET='fraxtil'

# Create an experiment dir under ddc/experiments/<your_x500>
EXP_DIR=os.path.join(SM_DIR, 'experiments', getpass.getuser())

if not os.path.exists(EXP_DIR):
    print("Making experiment directory: " + EXP_DIR)
    os.mkdir(EXP_DIR)


# In[ ]:


import random

exclude_pre_onsets_list = [True, False]
exclude_post_onsets_list = [True, False]
z_score_list = [True] # False

norm_type_list = ['layer', 'none'] # 'batch'
# do_attention_list = [True, False]
do_attention_list = [False]
do_transformer_list = [True, False]
# do_transformer_list = [True]
weight_strategy_list = ['rect', 'last'] # ['pos', 'seq', 'posbal']

dnn_nonlin_list = ['sigmoid', 'tanh', 'relu']
# dnn_nonlin_list = ['relu']
opt_list = ['sgd', 'adam', 'adagrad']
# opt_list = ['adagrad']
lr_list = [0.001, 0.01, 0.1] # Dropped 0.0001
# lr_list = [0.1]
lr_decay_rate_list = [0.1, 0.25, 0.5, 0.75, 0.95, 1.0]
# lr_decay_rate_list = [0.5, 0.95]
lr_decay_delay_list = [0, 2]

cnn_filter_shapes_list = ['7,3,10,3,3,20', '9,5,10,5,4,20', '3,3,16,3,3,32', '5,3,10,3,3,20', '5,5,32', '3,3,16,3,3,32,3,3,64', '7,2,16,5,2,32']
# cnn_filter_shapes_list = ['3,3,16,3,3,32,3,3,64']
cnn_pool_list = ['1,4,1,4', '2,4,2,4', '1,3,1,3', '2,2,2,2', '1,2,1,2', '2,2,1,1', '1,1,2,2', '2,2', '1,2,1,2,1,2']
# cnn_pool_list = ['2,2,1,1']
grad_clip_list = [1.0, 2.5, 5.0]
cnn_keep_prob_list = [0.7, 0.8, 0.9, 1.0]
dnn_keep_prob_list = [0.5] # Dropped 0.4, 0.6, 0.7, 0.8, 0.9, 1.0
input_keep_prob_list = [0.7, 0.8, 0.9, 1.0]
pooling_method_list = ['max', 'avg'] # , 'min'
# pooling_method_list = ['min']
d_model_list = [128, 256, 384, 512]
# d_model_list = [128]
num_transformer_layers_list = [1, 2, 4, 6]
num_heads_list = [1, 4, 8, 12]
# num_heads_list = [1]
ff_dim_list = [128, 256, 512, 1024, 2048]

rnn_keep_prob_list = [0.5] ### None of these matter if rnn_nlayers = 0?
# rnn_cell_type_list = ['rnn', 'gru', 'lstm']
rnn_cell_type_list = ['lstm'] # These don't matter if rnn_nlayers = 0?



def sample_config():
    return {
        'weight_strategy': random.choice(weight_strategy_list),
        'cnn_keep_prob': random.choice(cnn_keep_prob_list),
        'cnn_filter_shapes': random.choice(cnn_filter_shapes_list),
        'cnn_pool': random.choice(cnn_pool_list),
        'do_attention': random.choice(do_attention_list),
        'do_transformer': random.choice(do_transformer_list),
        'rnn_cell_type': random.choice(rnn_cell_type_list),
        'dnn_nonlin': random.choice(dnn_nonlin_list),
        'opt': random.choice(opt_list),
        'norm_type': random.choice(norm_type_list),
        'lr': random.choice(lr_list),
        'lr_decay_rate': random.choice(lr_decay_rate_list),
        'lr_decay_delay': random.choice(lr_decay_delay_list),
        'input_keep_prob': random.choice(input_keep_prob_list),
        'exclude_pre_onsets': random.choice(exclude_pre_onsets_list),
        'exclude_post_onsets': random.choice(exclude_post_onsets_list),
        'z_score': random.choice(z_score_list),
        'grad_clip': random.choice(grad_clip_list),
        'rnn_keep_prob': random.choice(rnn_keep_prob_list),
        'dnn_keep_prob': random.choice(dnn_keep_prob_list),
        'pooling_method': random.choice(pooling_method_list),
        'd_model': random.choice(d_model_list),
        'num_transformer_layers': random.choice(num_transformer_layers_list),
        'num_heads': random.choice(num_heads_list),
        'ff_dim': random.choice(ff_dim_list)
    }

num_runs = 500

for run_idx in range(num_runs):
    sampled = sample_config()
    
    run_id = (
        f"{sampled['weight_strategy']};{sampled['cnn_keep_prob']};{sampled['cnn_filter_shapes']};"
        f"{sampled['cnn_pool']};{sampled['do_attention']};{sampled['norm_type']};"
        f"{sampled['dnn_nonlin']};{sampled['z_score']};{sampled['opt']}/"
        f"{sampled['lr']};{sampled['lr_decay_rate']};{sampled['lr_decay_delay']};"
        f"{sampled['input_keep_prob']};{sampled['grad_clip']};{sampled['exclude_pre_onsets']};"
        f"{sampled['exclude_post_onsets']};{sampled['pooling_method']};{sampled['do_transformer']};"
        f"{sampled['d_model']};{sampled['num_transformer_layers']};{sampled['num_heads']};"
        f"{sampled['ff_dim']}"
    )

    training_args_dict = {
        'train_txt_fp': f'{SM_DIR}/data/chart_onset/{DATASET}/mel80hop441/{DATASET}_train.txt',
        'valid_txt_fp': f'{SM_DIR}/data/chart_onset/{DATASET}/mel80hop441/{DATASET}_valid.txt',
        'test_txt_fp': f'{SM_DIR}/data/chart_onset/{DATASET}/mel80hop441/{DATASET}_test.txt',
        'experiment_dir': EXP_DIR,
        'feat_diff_coarse_to_id_fp': f'{SM_DIR}/labels/{DATASET}/diff_coarse_to_id.txt',
        'audio_context_radius': 7,
        'audio_nbands': 80,
        'audio_nchannels': 3,
        'audio_select_channels': '0,1,2',
        'cnn_filter_shapes': sampled['cnn_filter_shapes'],
        'cnn_pool': sampled['cnn_pool'],
        'cnn_keep_prob': sampled['cnn_keep_prob'],
        'rnn_cell_type': sampled['rnn_cell_type'],
        'rnn_size': 200,
        'rnn_nlayers': 0,
        'rnn_nunroll': 1,
        'rnn_keep_prob': sampled['rnn_keep_prob'],
        'dnn_nonlin': sampled['dnn_nonlin'],
        'dnn_sizes': '256,128',
        'dnn_keep_prob': sampled['dnn_keep_prob'],
        'pooling_method': sampled['pooling_method'],
        'batch_size': 256,
        'weight_strategy': sampled['weight_strategy'],
        'exclude_onset_neighbors': 2,
        'grad_clip': sampled['grad_clip'],
        'opt': sampled['opt'],
        'norm_type': sampled['norm_type'],
        'lr': sampled['lr'],
        'lr_decay_rate': sampled['lr_decay_rate'],
        'lr_decay_delay': sampled['lr_decay_delay'],
        'nbatches_per_ckpt': 4000,
        'nbatches_per_eval': 4000,
        'nepochs': 5,
        'eval_window_type': 'hamming',
        'eval_window_width': 5,
        'eval_align_tolerance': 2,
        'z_score': sampled['z_score'],
        'exclude_pre_onsets': sampled['exclude_pre_onsets'],
        'exclude_post_onsets': sampled['exclude_post_onsets'],
#         'do_attention': sampled['do_attention'],
        'do_transformer': sampled['do_transformer'],
        'input_keep_prob': sampled['input_keep_prob'],
        'd_model': sampled['d_model'],
        'num_transformer_layers': sampled['num_transformer_layers'],
        'num_heads': sampled['num_heads'],
        'ff_dim': sampled['ff_dim']
        }


    training_args = parse_args(get_defaults=True)
    vars(training_args).update(training_args_dict)

    print(f"Starting run {run_idx+1}/{num_runs}: {run_id}")
    train_onset_model(training_args, OnsetModel=OnsetNet_Gina5)



# In[ ]:




