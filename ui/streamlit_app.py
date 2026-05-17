import streamlit as st
import pandas as pd
import sys
import os

# Add parent directory to path to allow importing analysis module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis import correlation

st.set_page_config(page_title="Correlation Heatmap Analytics", layout="wide")

st.title("Correlation Heatmap Analytics Tool")
st.markdown("Upload a CSV or Excel file to analyze correlations in your data.")

uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx'])

if uploaded_file is not None:
    try:
        filename, file_extension = os.path.splitext(uploaded_file.name)
        file_ext = file_extension.lower().lstrip('.')
        df = correlation.load_data(uploaded_file, file_ext)
        
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_cols:
            st.error("No numeric columns found in the dataset.")
        else:
            selected_cols = st.multiselect("Select columns for correlation analysis", numeric_cols, default=numeric_cols)
            
            if selected_cols:
                subset_df = df[selected_cols]
                corr_matrix = correlation.compute_correlation(subset_df)
                
                st.subheader("Correlation Analysis")
                
                tab1, tab2 = st.tabs(["Interactive Heatmap", "Static Heatmap"])
                
                with tab1:
                    fig_interactive = correlation.generate_interactive_heatmap(corr_matrix)
                    st.plotly_chart(fig_interactive, use_container_width=True)
                    
                with tab2:
                    fig_static = correlation.generate_static_heatmap(corr_matrix)
                    st.pyplot(fig_static)
                
                st.subheader("Insights")
                col1, col2 = st.columns(2)
                
                top_corrs = correlation.get_top_correlations(corr_matrix)
                
                with col1:
                    st.write("**Top Positive Correlations**")
                    st.dataframe(pd.DataFrame(top_corrs['top_positive']))
                    
                with col2:
                    st.write("**Top Negative Correlations**")
                    st.dataframe(pd.DataFrame(top_corrs['top_negative']))
                    
                st.subheader("High Correlations (> 0.7)")
                high_corrs = correlation.get_high_correlations(corr_matrix)
                if high_corrs:
                    st.dataframe(pd.DataFrame(high_corrs))
                    
                    # CSV Download
                    csv = pd.DataFrame(high_corrs).to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download High Correlations as CSV",
                        data=csv,
                        file_name='high_correlations.csv',
                        mime='text/csv',
                    )
                else:
                    st.info("No high correlations found above the threshold of 0.7.")
            else:
                st.warning("Please select at least one numeric column.")
                
    except Exception as e:
        st.error(f"An error occurred: {e}")
