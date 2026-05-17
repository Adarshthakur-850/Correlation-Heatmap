import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analysis.correlation import load_data, compute_correlation, get_high_correlations, get_top_correlations

def test_core_logic():
    print("Testing core logic...")
    
    # Create dummy data
    data = {
        'A': [1, 2, 3, 4, 5],
        'B': [2, 4, 6, 8, 10], # Perfectly correlated with A
        'C': [5, 4, 3, 2, 1], # Perfectly negatively correlated with A
        'D': [1, 1, 2, 2, 3]  # Weakly correlated
    }
    df = pd.DataFrame(data)
    
    # Save to CSV
    csv_path = 'test_data.csv'
    df.to_csv(csv_path, index=False)
    
    # Test load_data
    loaded_df = load_data(csv_path, 'csv')
    assert loaded_df.shape == df.shape
    print("load_data passed.")
    
    # Test compute_correlation
    corr_matrix = compute_correlation(loaded_df)
    print("Correlation Matrix:\n", corr_matrix)
    assert corr_matrix.loc['A', 'B'] > 0.99
    assert corr_matrix.loc['A', 'C'] < -0.99
    print("compute_correlation passed.")

    # Test get_top_correlations
    top_corrs = get_top_correlations(corr_matrix, top_n=2)
    print("Top Positive:\n", top_corrs['top_positive'])
    print("Top Negative:\n", top_corrs['top_negative'])
    
    # Verify strict separation
    for item in top_corrs['top_positive']:
        assert item['Correlation'] > 0, "Top positive contains non-positive values"
    for item in top_corrs['top_negative']:
        assert item['Correlation'] < 0, "Top negative contains non-negative values"
    print("get_top_correlations passed.")
    
    # Test get_high_correlations
    high_corrs = get_high_correlations(corr_matrix, threshold=0.7)
    print("High Correlations:\n", high_corrs)
    assert len(high_corrs) >= 2 # A-B and A-C should be there
    print("get_high_correlations passed.")
    
    # Clean up
    os.remove(csv_path)
    print("All core tests passed!")

if __name__ == "__main__":
    test_core_logic()
