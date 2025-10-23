import matplotlib
matplotlib.use('TkAgg')  # Set a compatible backend

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Load data from CSV files in the current directory
current_dir = os.getcwd()
lung_file = os.path.join(current_dir, 'Lung_segmentation_results.csv')
infection_file = os.path.join(current_dir, 'Infection_segmentation_results.csv')

if not os.path.exists(lung_file) or not os.path.exists(infection_file):
    raise FileNotFoundError("CSV files not found in the current directory.")

df_lung = pd.read_csv(lung_file)
df_infection = pd.read_csv(infection_file)

plt.figure(figsize=(18, 12))

# Bar plot for Accuracy
plt.subplot(2, 3, 1)
sns.barplot(x='Model', y='Accuracy', data=df_infection, hue='Model', dodge=False, legend=False)
plt.title('Infection - Accuracy')
plt.ylim(80, 100)

plt.subplot(2, 3, 4)
sns.barplot(x='Model', y='Accuracy', data=df_lung, hue='Model', dodge=False, legend=False)
plt.title('Lung - Accuracy')
plt.ylim(80, 100)

# Line plot for Loss
plt.subplot(2, 3, 2)
sns.lineplot(x='Model', y='Loss', data=df_infection, marker='o', color='b', label='Infection')
sns.lineplot(x='Model', y='Loss', data=df_lung, marker='o', color='r', label='Lung')
plt.title('Loss Comparison')
plt.legend()

# Pie chart for DSC
plt.subplot(2, 3, 3)
plt.pie(df_infection['DSC'], labels=df_infection['Model'], autopct='%1.1f%%', colors=sns.color_palette('pastel'), startangle=90)
plt.title('Infection - DSC')

plt.subplot(2, 3, 6)
plt.pie(df_lung['DSC'], labels=df_lung['Model'], autopct='%1.1f%%', colors=sns.color_palette('pastel'), startangle=90)
plt.title('Lung - DSC')

# Prepare data for box plot
df_infection['Task'] = 'Infection'
df_lung['Task'] = 'Lung'
combined_df = pd.concat([df_infection, df_lung])

# Box plot for Accuracy and Loss
plt.subplot(2, 3, 5)
sns.boxplot(x='Task', y='Accuracy', data=combined_df, hue='Task', dodge=False, legend=False)
sns.boxplot(x='Task', y='Loss', data=combined_df, hue='Task', dodge=False, legend=False)
plt.title('Accuracy and Loss Distribution')

plt.tight_layout()
plt.savefig('performance_dashboard.png', dpi=300)  # Save figure instead of showing

# Generate an image of the comparison table
fig, ax = plt.subplots(figsize=(8, 4))  # Create a new figure for the table
ax.axis('tight')
ax.axis('off')

# Combine the two dataframes and reset index
df_lung["Task"] = "Lung Segmentation"
df_infection["Task"] = "Infection Segmentation"
combined_table = pd.concat([df_lung, df_infection]).reset_index(drop=True)

# Convert table into an image format
table = ax.table(cellText=combined_table.values, colLabels=combined_table.columns, loc='center', cellLoc='center')

# Save the table as an image
plt.savefig('model_comparison_table.png', dpi=300, bbox_inches='tight')

plt.show()
