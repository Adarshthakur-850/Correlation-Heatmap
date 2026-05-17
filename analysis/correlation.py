import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

def load_data(file, file_type):
    """
    Load data from CSV or Excel file.
    
    Args:
        file: File object or path.
        file_type: str, 'csv' or 'xlsx'.
        
    Returns:
        pd.DataFrame: Loaded data.
    """
    try:
        if file_type == 'csv':
            df = pd.read_csv(file)
        elif file_type == 'xlsx':
            df = pd.read_excel(file)
        else:
            raise ValueError("Unsupported file type")
        return df
    except Exception as e:
        raise ValueError(f"Error loading file: {e}")

def compute_correlation(df):
    """
    Compute Pearson correlation matrix for numeric columns.
    
    Args:
        df: pd.DataFrame
        
    Returns:
        pd.DataFrame: Correlation matrix.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise ValueError("No numeric columns found in the dataset")
    return numeric_df.corr()

def get_high_correlations(corr_matrix, threshold=0.7):
    """
    Identify highly correlated pairs (> threshold).
    
    Args:
        corr_matrix: pd.DataFrame, correlation matrix.
        threshold: float, threshold for high correlation (absolute value).
        
    Returns:
        list of dict: List of highly correlated pairs with values.
    """
    # Create mask to ignore self-correlation and duplicates
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    corr_unstacked = corr_matrix.where(mask).stack().reset_index()
    corr_unstacked.columns = ['Variable 1', 'Variable 2', 'Correlation']
    
    # Filter by threshold
    high_corr = corr_unstacked[corr_unstacked['Correlation'].abs() > threshold]
    high_corr = high_corr.sort_values(by='Correlation', key=abs, ascending=False)
    
    return high_corr.to_dict('records')

def generate_static_heatmap(corr_matrix):
    """
    Generate a static heatmap using Seaborn.
    
    Args:
        corr_matrix: pd.DataFrame
        
    Returns:
        matplotlib.figure.Figure
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap')
    return plt.gcf()

def generate_interactive_heatmap(corr_matrix):
    """
    Generate an interactive heatmap using Plotly.
    
    Args:
        corr_matrix: pd.DataFrame
        
    Returns:
        plotly.graph_objects.Figure
    """
    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        origin="lower"
    )
    fig.update_layout(title="Interactive Correlation Heatmap")
    return fig

def get_top_correlations(corr_matrix, top_n=5):
    """
    Get top positive and negative correlations.
    """
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    corr_unstacked = corr_matrix.where(mask).stack().reset_index()
    corr_unstacked.columns = ['Variable 1', 'Variable 2', 'Correlation']
    
    # Filter for positive and negative
    positive_corrs = corr_unstacked[corr_unstacked['Correlation'] > 0]
    negative_corrs = corr_unstacked[corr_unstacked['Correlation'] < 0]

    top_positive = positive_corrs.sort_values(by='Correlation', ascending=False).head(top_n)
    top_negative = negative_corrs.sort_values(by='Correlation', ascending=True).head(top_n)
    
    return {
        "top_positive": top_positive.to_dict('records'),
        "top_negative": top_negative.to_dict('records')
    }
