import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from analysis.correlation import load_data, compute_correlation, get_high_correlations, get_top_correlations, generate_static_heatmap

def generate_sample_output():
    print("Generating sample output...")
    
    # Create output directory
    output_dir = 'sample_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dummy data
    np.random.seed(42)
    data = {
        'Feature_A': np.random.normal(0, 1, 100),
        'Feature_B': np.random.normal(0, 1, 100),
        'Feature_C': np.random.normal(0, 1, 100),
        'Feature_D': np.random.randint(0, 10, 100)
    }
    df = pd.DataFrame(data)
    
    # Add correlations
    df['Feature_B'] = df['Feature_A'] * 0.9 + np.random.normal(0, 0.1, 100) # High positive
    df['Feature_C'] = df['Feature_A'] * -0.8 + np.random.normal(0, 0.2, 100) # High negative
    
    # Save sample data
    df.to_csv(os.path.join(output_dir, 'sample_data.csv'), index=False)
    
    # Compute correlation
    corr_matrix = compute_correlation(df)
    corr_matrix.to_csv(os.path.join(output_dir, 'correlation_matrix.csv'))
    
    # High correlations
    high_corrs = get_high_correlations(corr_matrix)
    pd.DataFrame(high_corrs).to_csv(os.path.join(output_dir, 'high_correlations.csv'), index=False)
    
    # Top correlations
    top_corrs = get_top_correlations(corr_matrix)
    pd.DataFrame(top_corrs['top_positive']).to_csv(os.path.join(output_dir, 'top_positive_correlations.csv'), index=False)
    pd.DataFrame(top_corrs['top_negative']).to_csv(os.path.join(output_dir, 'top_negative_correlations.csv'), index=False)
    
    # Generate Heatmap
    fig = generate_static_heatmap(corr_matrix)
    fig.savefig(os.path.join(output_dir, 'heatmap.png'))
    
    print(f"Sample outputs generated in '{output_dir}' directory.")

if __name__ == "__main__":
    generate_sample_output()
