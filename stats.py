import pandas as pd
from scipy import stats
import numpy as np

# Load Excel file
df = pd.read_excel('data/experiment_results.xlsx')

# Define the column pairs you want to compare
column_pairs = [('FEMALE_COMPETENCE_N', 'MALE_COMPETENCE_N'), ('FEMALE_BENEVOLENCE_N', 'MALE_BENEVOLENCE_N'), ('FEMALE_INTEGRITY_N', 'MALE_INTEGRITY_N')]

# Initialize an empty list to store the results
results = []

# Loop through each pair of columns
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
        'Cohen\'s d': cohens_d
    })

# Convert the results list to a DataFrame
results_df = pd.DataFrame(results)

# Display the DataFrame
print(results_df)
