from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import io
import sys
import os

# Add parent directory to path to allow importing analysis module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.correlation import load_data, compute_correlation, get_high_correlations, get_top_correlations

app = FastAPI(title="Correlation Heatmap API")

@app.post("/heatmap")
async def create_heatmap(file: UploadFile = File(...)):
    try:
        content = await file.read()
        filename, file_extension = os.path.splitext(file.filename)
        file_ext = file_extension.lower().lstrip('.')
        
        if file_ext == 'csv':
            df = pd.read_csv(io.BytesIO(content))
        elif file_ext in ['xls', 'xlsx']:
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV or Excel file.")
            
        # Select numeric columns
        numeric_df = df.select_dtypes(include=['number'])
        if numeric_df.empty:
            raise HTTPException(status_code=400, detail="No numeric columns found in the dataset.")
            
        corr_matrix = numeric_df.corr()
        
        high_corrs = get_high_correlations(corr_matrix)
        top_corrs = get_top_correlations(corr_matrix)
        
        # Convert correlation matrix to JSON compatible format
        corr_json = corr_matrix.to_dict()
        
        return {
            "filename": file.filename,
            "columns": list(numeric_df.columns),
            "correlation_matrix": corr_json,
            "high_correlations": high_corrs,
            "top_correlations": top_corrs
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
