export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4


export CUDA_VISIBLE_DEVICES=3

python /exp/exp4/acp23xt/3rd_fp2_3.1/train.py&>trainoutput.log&
#python /exp/exp4/acp23xt/3rd_fp2_3.1/train.py

