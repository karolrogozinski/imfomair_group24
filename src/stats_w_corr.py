# Import necessary libraries
import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Excel file
df = pd.read_excel('data/experiment_results.xlsx').iloc[:-1]

# Define the column pairs for t-test and Cohen's d between female and male attributes
column_pairs = [('FEMALE_COMPETENCE', 'MALE_COMPETENCE'), 
                ('FEMALE_BENEVOLENCE', 'MALE_BENEVOLENCE'), 
                ('FEMALE_INTEGRITY', 'MALE_INTEGRITY')]

# Define the column pairs for correlation analysis within female and male attributes
female_pairs = [('FEMALE_COMPETENCE_N', 'FEMALE_BENEVOLENCE_N'), 
                ('FEMALE_COMPETENCE_N', 'FEMALE_INTEGRITY_N'), 
                ('FEMALE_BENEVOLENCE_N', 'FEMALE_INTEGRITY_N')]

male_pairs = [('MALE_COMPETENCE_N', 'MALE_BENEVOLENCE_N'), 
              ('MALE_COMPETENCE_N', 'MALE_INTEGRITY_N'), 
              ('MALE_BENEVOLENCE_N', 'MALE_INTEGRITY_N')]

# Initialize lists to store results
results = []
female_correlations = []
male_correlations = []

# Loop through each pair of columns for t-test and Cohen's d
for col1, col2 in column_pairs:
    group1 = df[col1].dropna()
    group2 = df[col2].dropna()
    
    # Perform independent t-test
    t_stat, p_value = stats.ttest_ind(group1, group2)
    
    # Calculate Cohen's d for effect size
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    
    # Cohen's d calculation
    cohens_d = (mean1 - mean2) / pooled_std
    
    # Append results for each pair as a dictionary
    results.append({
        'Column Pair': f'{col1} vs {col2}',
        'T-statistic': t_stat,
        'P-value': p_value,
        'Cohen\'s d': cohens_d,
    })

# Calculate correlations among female attributes
for col1, col2 in female_pairs:
    if col1 in df.columns and col2 in df.columns:
        valid_data = df[[col1, col2]].dropna()
        correlation, corr_p_value = stats.pearsonr(valid_data[col1], valid_data[col2])
        female_correlations.append({
            'Column Pair': f'{col1} vs {col2}',
            'Pearson Correlation': correlation,
            'P-value': corr_p_value,
        })

# Calculate correlations among male attributes
for col1, col2 in male_pairs:
    if col1 in df.columns and col2 in df.columns:
        valid_data = df[[col1, col2]].dropna()
        correlation, corr_p_value = stats.pearsonr(valid_data[col1], valid_data[col2])
        male_correlations.append({
            'Column Pair': f'{col1} vs {col2}',
            'Pearson Correlation': correlation,
            'P-value': corr_p_value,
        })

# Convert results lists to DataFrames
results_df = pd.DataFrame(results)
female_correlations_df = pd.DataFrame(female_correlations)
male_correlations_df = pd.DataFrame(male_correlations)

# Display the DataFrames
print("T-test and Cohen's d Results:")
print(results_df)
print("\nFemale Attribute Correlations:")
print(female_correlations_df)
print("\nMale Attribute Correlations:")
print(male_correlations_df)

# Plot a heatmap of the correlation matrix for female attributes
plt.figure(figsize=(6, 5))
female_corr_matrix = df[[col for col, _ in female_pairs] + [col2 for _, col2 in female_pairs]].corr()
sns.heatmap(female_corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Female Attributes Correlation Matrix')
plt.show()

# Plot a heatmap of the correlation matrix for male attributes
plt.figure(figsize=(6, 5))
male_corr_matrix = df[[col for col, _ in male_pairs] + [col2 for _, col2 in male_pairs]].corr()
sns.heatmap(male_corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Male Attributes Correlation Matrix')
plt.show()

# Plot scatter plots for each female attribute pair with trendlines
for col1, col2 in female_pairs:
    if col1 in df.columns and col2 in df.columns:
        plt.figure(figsize=(6, 4))
        sns.regplot(x=col1, y=col2, data=df, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.title(f'Scatter Plot with Trendline: {col1} vs {col2}')
        plt.show()

# Plot scatter plots for each male attribute pair with trendlines
for col1, col2 in male_pairs:
    if col1 in df.columns and col2 in df.columns:
        plt.figure(figsize=(6, 4))
        sns.regplot(x=col1, y=col2, data=df, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.title(f'Scatter Plot with Trendline: {col1} vs {col2}')
        plt.show()
