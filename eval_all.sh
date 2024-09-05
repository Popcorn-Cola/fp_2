export CUDA_VISIBLE_DEVICES=3

python eval_all.py -c 'logs' -t 10 -i 2 -g 'test/ground_truth' -z 'test/converted' -m 'WAVPDFMEL' -o 'resources/filelists/ljspeech/valid.txt'

#python eval_all.py -c 'logs/2024-06-17' -i 100 -g 'test_c_10steps/ground_truth' -z 'test_c_10steps/converted' -m 'WAVPDFMEL' -o 'resources/filelists/ljspeech/eval.txt'
