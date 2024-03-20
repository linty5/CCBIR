import numpy as np
import matplotlib.pyplot as plt

def plot_hist_and_scatter(neg_sets, pos_sets):
    plt.figure(figsize=(16, 6))

    ax1 = plt.gca()
    ax1.scatter(neg_sets, range(len(neg_sets)), color='red', label='Negative Samples (Scatter)', alpha=0.5)
    ax1.scatter(pos_sets, range(len(pos_sets)), color='green', label='Positive Samples (Scatter)', alpha=0.5)
    ax1.set_xlabel('Values')
    ax1.set_ylabel('Sample Index')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    bin_edges = np.linspace(-1, 1, 50)
    neg_hist, _ = np.histogram(neg_sets, bins=bin_edges, density=True)
    pos_hist, _ = np.histogram(pos_sets, bins=bin_edges, density=True)
    width = bin_edges[1] - bin_edges[0]
    ax2.bar(bin_edges[:-1], neg_hist, width=width, alpha=0.5, color='red', label='Negative Samples (Histogram)')
    ax2.bar(bin_edges[:-1], pos_hist, width=width, alpha=0.5, color='green', label='Positive Samples (Histogram)')
    ax2.set_ylabel('Frequency')
    ax2.legend(loc='upper left', bbox_to_anchor=(0, 0.9))  # Adjust the position of the legend slightly downward


    # plt.title('Scatter Plot and Histogram of Positive and Negative Samples')
    plt.grid(True)
    plt.savefig('./data/test/model_name/histogram_and_scatter.png')
    
def plot_cdf_and_scatter(neg_sets, pos_sets):
    neg_sets.sort()
    pos_sets.sort()

    neg_cdf = np.arange(1, len(neg_sets) + 1) / len(neg_sets)
    pos_cdf = np.arange(1, len(pos_sets) + 1) / len(pos_sets)

    plt.figure(figsize=(16, 6))

    plt.plot(neg_sets, neg_cdf, color='red', label='Negative Samples (CDF)')
    plt.plot(pos_sets, pos_cdf, color='green', label='Positive Samples (CDF)')
    plt.scatter(neg_sets, neg_cdf, color='red', alpha=0.5)
    plt.scatter(pos_sets, pos_cdf, color='green', alpha=0.5)

    plt.xlabel('Values')
    plt.ylabel('Cumulative Probability')
    plt.legend(loc='lower right')
    # plt.title('Cumulative Distribution Function (CDF) and Scatter Plot of Positive and Negative Samples')

    plt.tight_layout()
    plt.savefig('./data/test/model_name/cdf_and_scatter.png')
    

neg_sets = []
pos_sets = []
neg1_path = './data/test/model_name/neg1_score.txt'
neg2_path = './data/test/model_name/neg2_score.txt'
pos1_path = './data/test/model_name/pos1_score.txt'
pos2_path = './data/test/model_name/pos2_score.txt'

with open(neg1_path, 'r') as file:
    data = [float(line.strip()) for line in file]
    neg_sets.extend(data)
with open(neg2_path, 'r') as file:
    data = [float(line.strip()) for line in file]
    neg_sets.extend(data)

with open(pos1_path, 'r') as file:
    data = [float(line.strip()) for line in file]
    pos_sets.extend(data)
with open(pos2_path, 'r') as file:
    data = [float(line.strip()) for line in file]
    pos_sets.extend(data)

plot_hist_and_scatter(neg_sets, pos_sets)
# plot_cdf_and_scatter(neg_sets, pos_sets)
    
