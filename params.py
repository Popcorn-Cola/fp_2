# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

from model.utils import fix_len_compatibility

from datetime import datetime
current_date_time = datetime.now()
formatted_date = current_date_time.strftime("%Y-%m-%d")

# data parameters
train_filelist_path = 'resources/filelists/ljspeech/train.txt'
valid_filelist_path = 'resources/filelists/ljspeech/valid.txt'
test_filelist_path = 'resources/filelists/ljspeech/test.txt'
cmudict_path = 'resources/cmu_dictionary'
add_blank = True
n_feats = 80
n_spks = 1  # 247 for Libri-TTS filelist and 1 for LJSpeech
spk_emb_dim = 64
n_feats = 80
n_fft = 1024
sample_rate = 22050
hop_length = 256
win_length = 1024
f_min = 0
f_max = 8000

# encoder parameters
n_enc_channels = 192
filter_channels = 768
filter_channels_dp = 256
n_enc_layers = 6
enc_kernel = 3
enc_dropout = 0.1
n_heads = 2
window_size = 4

# decoder parameters
dec_dim = 64
beta_min = 0.05
beta_max = 20.0
pe_scale = 1000  # 1 for `grad-tts-old.pt` checkpoint
a=0
b=0
d=0         #d refers to dropout here
c=40         #c is for patch size
e=0         #e is for *n of batches    
l=1         #l=1, cutout in frequency domain. l=2, cutout in time domain.
p=0       #probability for time domain or frequency domain cutout

n_timesteps=10
# training parameters
log_dir="/exp/exp4/acp23xt/3rd_fp2_3.1/logs"
test_size = 4
n_epochs = 200 
batch_size = 16
learning_rate = 1e-4
seed = 20
save_every = 5
out_size = fix_len_compatibility(2*22050//256)
output_dir = 'save_0'
#output_dir = f'save_{n_epochs-1}'
