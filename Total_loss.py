import re
import matplotlib.pyplot as plt
import numpy as np
import argparse

def extract_numbers(line):
    numbers = re.findall(r'\d+\.\d+', line)
    duration_loss = round(float(numbers[0]), 3)
    prior_loss = round(float(numbers[1]), 3)
    diffusion_loss = round(float(numbers[2]), 3)
    return duration_loss, prior_loss, diffusion_loss

def read_lines(n, file_path):
    lines = []
    with open(file_path, 'r') as file:
        for i in range(n):
            line = file.readline()
            if not line:
                break
            lines.append(line.strip())
    return lines


parser = argparse.ArgumentParser()
parser.add_argument('-b', '--bl_training_log', type=str, required=True, help='file containing baseline model training log information')
parser.add_argument('-f', '--training_log', type=str, required=True, help='file containing training log information')
parser.add_argument('-s', '--save_location', type=str, required=True, help='position to save the files')
parser.add_argument('-n', '--n_lines', type=int, required=True, help='number of line to read from each file')
parser.add_argument('-itv', '--interval', type=int, required=True, help='interval of recorded loss information in training log file')
args = parser.parse_args()

training_log = args.training_log
bl_training_log = args.bl_training_log

save_location = args.save_location
itv = args.interval

cv_lines = read_lines(args.n_lines, training_log)
bl_lines = read_lines(args.n_lines, bl_training_log)
duration_losses = []
prior_losses = []
diffusion_losses = []
Total_losses = []

bl_duration_losses = []
bl_prior_losses = []
bl_diffusion_losses = []
bl_Total_losses = []

for line in cv_lines:
    numbers = extract_numbers(line)
    duration_loss, prior_loss , diffusion_loss  = numbers
    duration_losses.append(duration_loss)
    prior_losses.append(prior_loss)
    diffusion_losses.append(diffusion_loss)
    Total_losses.append(f'{prior_loss+diffusion_loss+duration_loss}')

for line in bl_lines:
    numbers = extract_numbers(line)
    duration_loss, prior_loss , diffusion_loss  = numbers
    bl_duration_losses.append(duration_loss)
    bl_prior_losses.append(prior_loss)
    bl_diffusion_losses.append(diffusion_loss)
    bl_Total_losses.append(f'{prior_loss+diffusion_loss+duration_loss}')


fig, ax = plt.subplots()
ax.plot(diffusion_losses, color='red', label='decoder loss with masking noise')

# Set x-axis labels to appear every 4 data points
xticks = np.arange(1, len(diffusion_losses)+1, 20)
ax.set_xticks(xticks)


# Optional: Rotate the x labels for better readability
plt.xticks(rotation=45)

# Add labels and title
ax.set_xlabel('Epoch')
ax.set_title('Training_loss')

plt.savefig(f'{save_location}')




#plt.plot(duration_losses, color='blue', label='duration loss')
#plt.plot(prior_losses, color='green', label='prior loss')
#plt.plot(diffusion_losses, color='red', label='decoder loss with masking noise')
#plt.plot(Total_losses, color='orange', label='Total loss')
#plt.plot(bl_diffusion_losses, color='green', label='diffusion loss of baseline model')


#plt.xlabel('Epoch')
#plt.title('Training_loss')

#plt.legend()

#plt.savefig(f'{save_location}')
