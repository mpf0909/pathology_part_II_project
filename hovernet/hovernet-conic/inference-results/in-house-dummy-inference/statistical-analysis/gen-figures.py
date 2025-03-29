
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA

def plot_combined_barchart(normal_csv, coeliac_csv, normal_json, coeliac_json, output_file_name):
    normal_df = pd.read_csv(normal_csv)
    coeliac_df = pd.read_csv(coeliac_csv)

    with open(normal_json, 'r') as f:
        normal_metadata = json.load(f)
    with open(coeliac_json, 'r') as f:
        coeliac_metadata = json.load(f)
    
    print(normal_metadata)
    print(coeliac_metadata)
    
    total_normal_cells = normal_df.sum(axis=0)
    total_coeliac_cells = coeliac_df.sum(axis=0)

    area_normalised_normal_cells = total_normal_cells/
    coeliac_averages = coeliac_df.mean()
    
    combined_df = pd.DataFrame({
        'Cell Type': normal_averages.index,
        'Normal': normal_averages.values,
        'Coeliac': coeliac_averages.values
    })
    
    combined_df = pd.melt(combined_df, id_vars=['Cell Type'], var_name='Condition', value_name='Average Count')
    
    plt.figure(figsize=(8, 5))
    sns.barplot(data=combined_df, x='Cell Type', y='Average Count', hue='Condition', palette=['skyblue', 'salmon'])
    plt.xlabel('Cell Type')
    plt.ylabel('Average Count')
    plt.title('Average Number of Each Cell Type (Normal vs Coeliac) per patch')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig(output_file_name, dpi=300, bbox_inches='tight')
    plt.close()

def plot_combined_boxplots(normal_csv, coeliac_csv, output_file_name):
    normal_df = pd.read_csv(normal_csv)
    coeliac_df = pd.read_csv(coeliac_csv)
    
    normal_df['Condition'] = 'Normal'
    coeliac_df['Condition'] = 'Coeliac'
    
    df = pd.concat([normal_df, coeliac_df], ignore_index=True)
    
    df_melted = df.melt(id_vars=['Condition'], var_name='Cell Type', value_name='Count')
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Box-and-Whisker Plots for Each Cell Type (Normal vs Coeliac)', fontsize=16)
    axes = axes.flatten()

    cell_types = df_melted['Cell Type'].unique()
    for ax, cell_type in zip(axes, cell_types):
        sns.boxplot(data=df_melted[df_melted['Cell Type'] == cell_type], x='Condition', y='Count', ax=ax, palette=['skyblue', 'salmon'], showfliers=True)
        ax.set_title(f"average numer of {cell_type} per patch")
        ax.set_xlabel('')
        ax.set_ylabel('Count')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plt.savefig(output_file_name, dpi=300, bbox_inches='tight')
    plt.close()

plot_combined_boxplots(
    normal_csv='../toy_data_20x_inference/normal/225363019/valid_pred_cell.csv', 
    coeliac_csv='../toy_data_20x_inference/coeliac/235364597/valid_pred_cell.csv', 
    output_file_name='combined_tissue_boxplots.png'
)

plot_combined_barchart(
    normal_csv='../toy_data_20x_inference/normal/225363019/valid_pred_cell.csv', 
    coeliac_csv='../toy_data_20x_inference/coeliac/235364597/valid_pred_cell.csv', 
    normal_json = 
    coeliac_json = 
    output_file_name='combined_tissue_barchart.png'
)

"""
### logistic regression considering nuclei class ###

# Load datasets
normal_df = pd.read_csv('../toy_data_20x_inference/normal/225363019/valid_pred_cell.csv')
coeliac_df = pd.read_csv('../toy_data_20x_inference/coeliac/235364597/valid_pred_cell.csv')

# Assign labels: 0 for normal, 1 for coeliac
normal_df['label'] = 0
coeliac_df['label'] = 1

# Combine datasets
combined_df = pd.concat([normal_df, coeliac_df], ignore_index=True)

# Define features and labels
X = combined_df.drop(columns=['label'])
y = combined_df['label']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train logistic regression model
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Performance metrics
print('Accuracy:', accuracy_score(y_test, y_pred))
print('ROC-AUC:', roc_auc_score(y_test, y_pred_proba))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create a dataframe with the PCA results
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['label'] = y

# Plot the first two principal components
plt.figure(figsize=(8, 6))
plt.scatter(pca_df[pca_df['label'] == 0]['PC1'], pca_df[pca_df['label'] == 0]['PC2'], 
            c='skyblue', alpha=0.7, s=10, label='Normal')
plt.scatter(pca_df[pca_df['label'] == 1]['PC1'], pca_df[pca_df['label'] == 1]['PC2'], 
            c='salmon', alpha=0.7, s=10, label='Coeliac')
plt.title('PCA of Nuclei Classes in Normal vs Coeliac Disease Images')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Condition')
plt.savefig("pca_plot.png")
plt.close()


"import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

### logistic regression just considering total number of nuclei ###

normal_df = pd.read_csv('../toy_data_20x_inference/normal/225363019/valid_pred_cell.csv')
coeliac_df = pd.read_csv('../toy_data_20x_inference/coeliac/235364597/valid_pred_cell.csv')

# Sum all nuclei counts per image
normal_df['total_nuclei'] = normal_df.sum(axis=1)
coeliac_df['total_nuclei'] = coeliac_df.sum(axis=1)

# Add labels: 0 for normal, 1 for coeliac
normal_df['label'] = 0
coeliac_df['label'] = 1

# Combine datasets
combined_df = pd.concat([normal_df, coeliac_df], ignore_index=True)

# Prepare features and labels
X = combined_df[['total_nuclei']]
y = combined_df['label']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Logistic Regression Model
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate performance
print('Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Output model coefficients
print('Intercept:', model.intercept_)
print('Coefficient:', model.coef_[0])
"""