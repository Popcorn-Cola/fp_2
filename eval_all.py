import argparse
import json
import re
import torch
import params
import shutil
import numpy as np
from model import GradTTS
from torch.utils.data import DataLoader
from data import TextMelDataset, TextMelBatchCollate
from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils import intersperse
from scipy.io.wavfile import write
from scipy import integrate

import matplotlib.pyplot as plot
import sys, os

sys.path.append('./hifi-gan/')
from env import AttrDict
from models import Generator as HiFiGAN


cmudict_path = params.cmudict_path
add_blank = params.add_blank
n_fft = params.n_fft
sample_rate = params.sample_rate
hop_length = params.hop_length
win_length = params.win_length
f_min = params.f_min
f_max = params.f_max
n_feats = params.n_feats
out_size = params.out_size
n_timesteps = params.n_timesteps
a = params.a
b = params.b
c = params.c
d = params.d
e = params.e
l = params.l
p = params.p

HIFIGAN_CONFIG = './checkpts/hifigan-config.json'
HIFIGAN_CHECKPT = './checkpts/hifigan.pt'

def _get_basename(path: str) -> str:
    return os.path.splitext(os.path.split(path)[-1])[0]

def save_mel_spectrograms_to_file(mel_spectrograms, output_dir): # input a list of mel tensors, output one mel tensor to one file in output_dir
    for i, mel_spec in enumerate(mel_spectrograms):

        torch.save(mel_spec , f'{output_dir}/mel_{i}.pt')

def pt_to_pdf(pt, pdf, vmin=-12.5, vmax=0.0):
    spec = pt
    fig = plot.figure(figsize=(20, 4), tight_layout=True)
    subfig = fig.add_subplot()
    image = subfig.imshow(spec, cmap="jet", origin="lower", aspect="equal", interpolation="none", vmax=vmax,
                          vmin=vmin)
    fig.colorbar(mappable=image, orientation='vertical', ax=subfig, shrink=0.5)
    plot.savefig(pdf, format="pdf")
    plot.close()

def get_integer_part(s):
    return int(''.join(filter(str.isdigit, s)))

###################################################################################################
def compute_integral(a, b, c, d, n):
    # Define the integrand function
    def integrand(s):
        exp_part = np.exp(2 * (a * n + b))
        term1 = 1 - (1 / (c * s + d))

        # Define the inner integral function
        def inner_integral(u):
            return 1 - (1 / (c * u + d))

        # Compute the inner integral from s to n
        inner_integral_value, _ = integrate.quad(inner_integral, s, n)
        inner_exp_part = np.exp(-2 * inner_integral_value)

        return exp_part * term1 * inner_exp_part

    # Perform the outer integration from 0 to n
    result, error = integrate.quad(integrand, 0, n)

    # Multiply by 2 as per the given expression
    return 2 * result

#print('Initializing Variance Parameter')
#n_list = list(range(n_timesteps + 1))
#std_list = torch.tensor([np.sqrt( compute_integral(a, b, c, d, n) ) for n in n_list]).to('cuda')

#####################################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint_dir', type=str, required=True,
                        help='path to checkpoint directory of Grad-TTS')
    parser.add_argument('-t', '--n_timesteps', type=int, required=False, default=10,
                        help='number of timesteps of reverse diffusion')
    parser.add_argument('-g', '--gt_dir', type=str, required=False, default='eval/original',
                        help='location to save the ground truth data')
    parser.add_argument('-z', '--cvt_dir', type=str, required=False, default='eval/converted',
                        help='location to save the converted data')
    parser.add_argument('-o', '--original', type=str, required=False, default='',
                        help='location of the .wav data to be evaluated/tested')
    parser.add_argument('-i', '--epoch_interval', type=int, required=False, default=100,
                        help='The interval between epochs to be evaluated')
    parser.add_argument('-m', '--evaluation_mode', type=str, required=False, default='WAVPDFMEL_ENCODER',
                        help='WAVPDFMEL  or LOSSES or WAVPDFMEL_ENCODER')
    args = parser.parse_args()

    gt_dir = args.gt_dir
    cvt_dir = args.cvt_dir
    checkpoint_dir = args.checkpoint_dir
    epoch_interval = args.epoch_interval
    evaluation_mode = args.evaluation_mode
    original = args.original
    n_timesteps = args.n_timesteps

    # get cmu dictionary
    # importcmudict
    cmu = cmudict.CMUDict('./resources/cmu_dictionary')

    # import cmudict
    print('Logging validation/test dataset...')
    valid_dataset = TextMelDataset(original, cmudict_path, add_blank,
                                  n_fft, n_feats, sample_rate, hop_length,
                                  win_length, f_min, f_max)

    print('get checkpts paths')
    checkpoint_files = [os.path.join(checkpoint_dir, file) for file in os.listdir(checkpoint_dir) if
                        file.endswith('.pt')]
    checkpoint_files = sorted(checkpoint_files, key=get_integer_part)

    print('Initialize GradTTS MODEL')
    generator = GradTTS(len(symbols) + 1, params.n_spks, params.spk_emb_dim,
                        params.n_enc_channels, params.filter_channels,
                        params.filter_channels_dp, params.n_heads, params.n_enc_layers,
                        params.enc_kernel, params.enc_dropout, params.window_size,
                        params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale, a, b, c, d, e ,l, p, n_timesteps, None).cuda()

    print('build Valid batch...')
    # idx = np.random.choice(list(range(len(test_dataset))), size=params.test_size, replace=False)

    valid_batch_text = []
    valid_batch_mel = []
    filepaths = []
    for filepath_and_text in valid_dataset.filepaths_and_text:
        # Each entry of filepaths_and_text is a [filepath, text_content] list.
        filepath, text = filepath_and_text[0], filepath_and_text[1]
        mel = valid_dataset.get_mel(filepath)

        filepaths.append(filepath)
        valid_batch_text.append(text)  # [{'y': mel, 'x': text}, {'y': mel, 'x': text}, {'y': mel, 'x': text}]
        valid_batch_mel.append(mel)


    if evaluation_mode == 'WAVPDFMEL' or evaluation_mode == 'WAVPDFMEL_ENCODER':

        print('output original mel spectrogram and text')
        print('all mel spectrogram is written in a file')
        print('all text is written in a file')
        if not os.path.exists(gt_dir):
            os.makedirs(f'{gt_dir}')
        if not os.path.exists(cvt_dir):
            os.makedirs(f'{cvt_dir}')

        gt_text = gt_dir+'/text.txt'
        with open(gt_text, 'w') as text_file:
            for i, item in enumerate(valid_batch_text):
                text_file.write(f"{valid_batch_text[i]}\n")

        texts = valid_batch_text

        print('move the.wav file from LJSpeech dataset to evaluation/test directory')
        for i, filepath in enumerate(filepaths):
            shutil.copy(filepath, f'{gt_dir}/output_{i}.wav')

        save_mel_spectrograms_to_file(valid_batch_mel, f'{gt_dir}')
        
        for i, mel_spec in enumerate(valid_batch_mel):
            pt_to_pdf(mel_spec, f'{gt_dir}/mel_{i}.pdf' , vmin=-12.5, vmax=0.0)

        print('Initializing HiFi-GAN as vocoder')
        
        with open(HIFIGAN_CONFIG) as f:
            h = AttrDict(json.load(f))
        vocoder = HiFiGAN(h)
        vocoder.load_state_dict(torch.load(HIFIGAN_CHECKPT, map_location=lambda loc, storage: loc)['generator'])
        _ = vocoder.cuda().eval()
        vocoder.remove_weight_norm()

        if evaluation_mode == 'WAVPDFMEL':
            for i in range(0, len(checkpoint_files),epoch_interval):
                y_mel = []
                #get integer parts of the file
                checkpoint_name = _get_basename(checkpoint_files[i])
                number_part = get_integer_part(checkpoint_name)
                index = int(number_part)

                generator.load_state_dict(torch.load(f'{checkpoint_files[i]}', map_location=lambda loc, storage: loc))
                _ = generator.cuda().eval()
                print(f'Doing the {index}st epoch')
                if not os.path.exists(f'{cvt_dir}/Epoch_{index}'):
                    os.makedirs(f'{cvt_dir}/Epoch_{index}')
                with torch.no_grad():
                    for i, text in enumerate(texts):
                            # convert word to phonemes:
                            x = torch.LongTensor(intersperse(text_to_sequence(text, dictionary=cmu), len(symbols))).cuda()[None]
                            x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
                            y_enc, y_dec, attn = generator.forward(x, x_lengths, n_timesteps=n_timesteps, temperature=1.5,
                                                                   stoc=False, length_scale=0.91)
                            
                            if torch.isnan(y_dec).any():
                                print("Warning: The tensor contains NaN")
                            if torch.isinf(y_dec).any():
                                print("Warning: The tensor contains infinite values.")
                            
                            y_mel.append(y_dec) # [put all mel tensor from text file into a list]
                            
                            audio = (vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)
                            write(f'{cvt_dir}/Epoch_{index}/output_{i}.wav', 22050, audio)
                            pt_to_pdf(y_dec.cpu().squeeze(0), f'{cvt_dir}/Epoch_{index}/dec_{i}.pdf')
                            pt_to_pdf(y_enc.cpu().squeeze(0), f'{cvt_dir}/Epoch_{index}/enc_{i}.pdf')
                save_mel_spectrograms_to_file(y_mel, f'{cvt_dir}/Epoch_{index}')

        elif evaluation_mode == 'WAVPDFMEL_ENCODER':
            for i in range(0,len(checkpoint_files),epoch_interval):
                y_mel = []
                #get integer parts of the file
                checkpoint_name = _get_basename(checkpoint_files[i])
                number_part = get_integer_part(checkpoint_name)
                index = int(number_part)

                generator.load_state_dict(torch.load(f'{checkpoint_files[i]}', map_location=lambda loc, storage: loc))
                _ = generator.cuda().eval()
                print(f'Doing the {index}st epoch')
                os.makedirs(f'{cvt_dir}/Epoch_{index}')
                with torch.no_grad():
                    for i, text in enumerate(texts):
                            # convert word to phonemes:
                            x = torch.LongTensor(intersperse(text_to_sequence(text, dictionary=cmu), len(symbols))).cuda()[None]
                            x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
                            y_enc, y_dec, attn = generator.forward(x, x_lengths, n_timesteps=args.timesteps, temperature=1.5,
                                                                   stoc=False, length_scale=0.91)
                            y_mel.append(y_enc) # [put all mel tensor from text file into a list]

                            audio = (vocoder.forward(y_enc).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)
                            write(f'{cvt_dir}/Epoch_{index}/output_{i}.wav', 22050, audio)
                            # pt_to_pdf(y_dec.cpu().squeeze(0), f'{cvt_dir}/Epoch_{index}/dec_{i}.pdf')
                            # pt_to_pdf(y_enc.cpu().squeeze(0), f'{cvt_dir}/Epoch_{index}/enc_{i}.pdf')
                save_mel_spectrograms_to_file(y_mel, f'{cvt_dir}/Epoch_{index}')




    if evaluation_mode == 'LOSSES':
        batch_size = len(filepaths)//20
        print(f'length of valid list is {batch_size}')
        batch_collate = TextMelBatchCollate()
        loader = DataLoader(dataset=valid_dataset, batch_size=batch_size,
                            collate_fn=batch_collate, drop_last=True,
                            num_workers=4, shuffle=False)

        for i in range(0, len(checkpoint_files), epoch_interval):
            dur_loss_list = []
            prior_loss_list = []
            diff_loss_list = []
            for n , valid_data in enumerate(loader):
                # get integer parts of the file
                checkpoint_name = _get_basename(checkpoint_files[i])
                number_part = get_integer_part(checkpoint_name)
                index = int(number_part)

                generator.load_state_dict(torch.load(f'{checkpoint_files[i]}', map_location=lambda loc, storage: loc))
                _ = generator.cuda().eval()
                print(f'Doing the {index}st epoch')

                x, x_lengths = valid_data['x'].cuda(), valid_data['x_lengths'].cuda()
                y, y_lengths = valid_data['y'].cuda(), valid_data['y_lengths'].cuda()
                print(f'size of x is {np.shape(x)} , size of x_lengths is {np.shape(x_lengths)}')
                print(f'size of y is {np.shape(y)}, size of y_lengths is {np.shape(y_lengths)}')

                dur_loss, prior_loss, diff_loss = generator.compute_loss(x, x_lengths,y, y_lengths)
                dur_loss_list.append(dur_loss.item())
                prior_loss_list.append(prior_loss.item())
                diff_loss_list.append(diff_loss.item())

            dur_loss = np.mean(dur_loss_list)
            prior_loss = np.mean(prior_loss_list)
            diff_loss = np.mean(diff_loss_list)

            loss = sum([dur_loss, prior_loss, diff_loss])
            log_msg = 'Epoch %d: duration loss = %.3f ' % (index, np.mean(dur_loss))
            log_msg += '| prior loss = %.3f ' % np.mean(prior_loss)
            log_msg += '| diffusion loss = %.3f' % np.mean(diff_loss)
            log_msg += '| Total loss = %.3f\n' % (np.mean(dur_loss) + np.mean(prior_loss) + np.mean(diff_loss))

            with open('train.log', 'a') as f:
                f.write(log_msg)
