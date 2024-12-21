import numpy as np
from scipy import stats
import pandas as pd

# Create dataframe with the results
data = {
    'Model': ['Random Forest', 'Gradient Boosting', 'Support Vector Machine', 
              'Fully Connected Neural Network'],
    'Audio_Only': [58.65, 57.89, 61.43, 61.94],
    'Audio_Social': [81.29, 80.45, 77.44, 81.26],
    'MSE_Audio': [0.4135, 0.4211, 0.3835, 0.3806],
    'MSE_Combined': [0.1880, 0.1955, 0.2226, 0.1874]
}

df = pd.DataFrame(data)

def perform_statistical_analysis(df):
    # Calculate improvements
    df['Accuracy_Improvement'] = df['Audio_Social'] - df['Audio_Only']
    df['MSE_Improvement'] = df['MSE_Audio'] - df['MSE_Combined']
    
    # Perform paired t-test for accuracy
    t_stat_acc, p_value_acc = stats.ttest_rel(df['Audio_Social'], df['Audio_Only'])
    
    # Perform paired t-test for MSE
    t_stat_mse, p_value_mse = stats.ttest_rel(df['MSE_Audio'], df['MSE_Combined'])
    
    # Calculate effect sizes (Cohen's d)
    def cohens_d(group1, group2):
        diff = group1 - group2
        n = len(group1)
        s = np.sqrt(((n-1) * np.var(group1, ddof=1) + 
                     (n-1) * np.var(group2, ddof=1)) / (2*n - 2))
        return np.mean(diff) / s
    
    d_acc = cohens_d(df['Audio_Social'], df['Audio_Only'])
    d_mse = cohens_d(df['MSE_Audio'], df['MSE_Combined'])
    
    # Print results
    print("\nStatistical Analysis Results:")
    print("-" * 50)
    
    print("\nAccuracy Analysis:")
    print(f"Mean Improvement: {df['Accuracy_Improvement'].mean():.2f}%")
    print(f"T-statistic: {t_stat_acc:.4f}")
    print(f"P-value: {p_value_acc:.4f}")
    print(f"Cohen's d: {d_acc:.4f}")
    
    print("\nMSE Analysis:")
    print(f"Mean MSE Reduction: {df['MSE_Improvement'].mean():.4f}")
    print(f"T-statistic: {t_stat_mse:.4f}")
    print(f"P-value: {p_value_mse:.4f}")
    print(f"Cohen's d: {d_mse:.4f}")
    
    # Print detailed improvements by model
    print("\nModel-wise Improvements:")
    print("-" * 50)
    for i, row in df.iterrows():
        print(f"\n{row['Model']}:")
        print(f"Accuracy: {row['Accuracy_Improvement']:.2f}% improvement")
        print(f"MSE: {row['MSE_Improvement']:.4f} reduction")

# Run the analysis
perform_statistical_analysis(df)