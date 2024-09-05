export CUDA_VISIBLE_DEVICES=3

source /store/store4/software/bin/anaconda3/etc/profile.d/conda.sh
conda activate Grad-TTS-EVAL 

root_directory='/exp/exp4/acp23xt/3rd_fp_3.1'

Gt_directory="${root_directory}/test/ground_truth"
Converted_directory="${root_directory}/test/converted" #it containes N epoch subdirectories, each comparing with Ground truth directory
Output_directory="${root_directory}/test/metrics"
if [ ! -d "$Output_directory" ]; then
    mkdir -p "$Output_directory"
fi

#*******************************************************************************
################################################################################
#*******************************************************************************


#When compute MCD of a single wave file

#python "${root_directory}/tools/MCD/MCD.py" --gt_wavdir_or_wavscp "${Gt_directory}/output_0.wav" --gen_wavdir_or_wavscp  "${Converted_directory}/Epoch_0/output_0.wav" --outdir "$Output_directory"



#*******************************************************************************
################################################################################
#*******************************************************************************


#When dealing with F0 of a folder containing all wavefiles, the folder's name are passed to gen_wavdir_or_wavscp parameter 

#rm  "$Output_directory/Epoch_151/mean_mcd.txt"
#rm  "$Output_directory/Epoch_151/std_mcd.txt"
#python "${root_directory}/tools/MCD/MCD.py" --gt_wavdir_or_wavscp "$Gt_directory" --gen_wavdir_or_wavscp "$Converted_directory/Epoch_151"  --outdir "$Output_directory/Epoch_151"


#*******************************************************************************
################################################################################
#*******************************************************************************


#When dealing with F0 of a folder containing subfolders

sorted_folder=$(python "${root_directory}/tools/MCD/sorted.py" -d $Converted_directory) #sort the epoch directory according to epoch numbers

folder_list=($sorted_folder)

length=${#folder_list[@]}

rm  "$Output_directory/mean_mcd.txt"
rm  "$Output_directory/std_mcd.txt"

# Loop through the folder_list with an interval of 10
for ((i=0; i<length; i+=1)); do
    folder=${folder_list[$i]}
    echo "$folder"
    python "${root_directory}/tools/MCD/MCD.py" --gt_wavdir_or_wavscp "$Gt_directory" --gen_wavdir_or_wavscp "$folder" --outdir "$Output_directory"
done


python "${root_directory}/tools/MCD/line_chart.py" -p $Output_directory/mean_mcd.txt -s 0 -i 10 -t MCD -x Epoch -y mean_mcd -o $Output_directory/mean_mcd.png

python "${root_directory}/tools/MCD/line_chart.py" -p $Output_directory/std_mcd.txt -s 0 -i 10 -t MCD -x Epoch -y std_mcd -o $Output_directory/std_mcd.png

