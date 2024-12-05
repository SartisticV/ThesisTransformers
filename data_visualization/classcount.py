import json
import matplotlib.pyplot as plt
import numpy as np

# ID-to-class dictionary
id_to_class = {1: 'pedestrian', 2: 'people', 3: 'bicycle', 4: 'car', 5: 'van', 6: 'truck', 7: 'tricycle', 8: 'awning-tricycle', 9: 'bus', 10: 'motor', 11: 'others'}

# Define size thresholds
small_thresh = 32 * 32    # 1024 sq pixels
medium_thresh = 96 * 96   # 9216 sq pixels
large_thresh = 1e5 * 1e5  # 1e10 sq pixels (arbitrary large)

# Function to categorize object size based on area
def categorize_size(bbox):
    width, height = bbox[2], bbox[3]
    area = width * height
    if area <= small_thresh:
        return 'small'
    elif area <= medium_thresh:
        return 'medium'
    else:
        return 'large'

# Initialize counts for each category and size (for train and test sets)
train_counts = {'small': [0] * len(id_to_class), 'medium': [0] * len(id_to_class), 'large': [0] * len(id_to_class)}
train_total = {'small': 0, 'medium': 0, 'large': 0}
test_counts = {'small': [0] * len(id_to_class), 'medium': [0] * len(id_to_class), 'large': [0] * len(id_to_class)}
test_total = {'small': 0, 'medium': 0, 'large': 0}

# Process training set
with open('./visdrone_new/train/labels.json', 'r') as fp:
    train = json.load(fp)

for a in train['annotations']:
    category_id = a['category_id'] - 1  # Adjust for zero indexing
    size_category = categorize_size(a['bbox'])  # Determine the size category
    train_counts[size_category][category_id] += 1
    train_total[size_category] += 1

# Process testing set
with open('./visdrone_new/val/labels.json', 'r') as fp:
    test = json.load(fp)

for a in test['annotations']:
    category_id = a['category_id'] - 1  # Adjust for zero indexing
    size_category = categorize_size(a['bbox'])  # Determine the size category
    test_counts[size_category][category_id] += 1
    test_total[size_category] += 1

print('train: ', train_total)
print('test: ', test_total)
# Distinct colors for each class
colors = plt.cm.get_cmap('tab20', 11).colors

fig, axs = plt.subplots(3, 2, figsize=(12, 18), sharex=False)

# Function to create bar plots for each size category
def plot_bars(ax, size_category, counts, title):
    ax.bar(id_to_class.values(), counts, color=colors)
    ax.set_title(f'{title} ({size_category.capitalize()} Objects)')
    ax.set_ylabel('Count')
    ax.set_xlabel('Class')
    ax.set_xticks([])

# Plot training and testing data for small, medium, and large objects
plot_bars(axs[0, 0], 'small', train_counts['small'], 'Training Set')
plot_bars(axs[0, 1], 'small', test_counts['small'], 'Testing Set')
plot_bars(axs[1, 0], 'medium', train_counts['medium'], 'Training Set')
plot_bars(axs[1, 1], 'medium', test_counts['medium'], 'Testing Set')
plot_bars(axs[2, 0], 'large', train_counts['large'], 'Training Set')
plot_bars(axs[2, 1], 'large', test_counts['large'], 'Testing Set')

# Customizing the legend
handles = [plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(11)]
legend_labels = id_to_class.values()

# Customizing the legend below the plot
fig.legend(handles, legend_labels, loc='lower center', ncol=11, columnspacing=0.3, fontsize='large')

# Adjust layout
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()

