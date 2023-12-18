import numpy as np

filename = "final_fids_2048.txt"
with open(filename, 'r') as file:
    fids = np.array([float(x) for x in file.read().split(' ')])

# Calculate mean and standard deviation
mean = np.mean(fids)
std_dev = np.std(fids)

# Set the confidence level (e.g., 95%)
confidence_level = 0.95

# Calculate the standard error
standard_error = std_dev / np.sqrt(len(fids))

# Calculate the margin of error
margin_of_error = standard_error * 1.96 # Using a z-score of 1.96 for 95% confidence

# Calculate the confidence interval
lower_bound = mean - margin_of_error
upper_bound = mean + margin_of_error

print(f"Confidence Interval: {round(mean, 4)} +/- {round(margin_of_error, 4)}")
