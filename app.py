import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
import os
from backend.kmeans import run_kmeans
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})
# Read the first CSV file as a DataFrame
dataINC5000 = pd.read_csv('C:\\Users\\arhod\\Desktop\\Diploma-vscode\\INC 5000 Companies 2019.csv')

# Count the size of the DataFrames in MB
size_inc5000_mb = dataINC5000.memory_usage(deep=True).sum() / (1024 * 1024)

'''
print(f"First CSV data size: {size_indian_faculty_mb:.2f} MB")
print(f"Second CSV data size: {size_inc5000_mb:.2f} MB")
'''

dataINC5000 = dataINC5000.replace({np.nan: None})


@app.post("/api/upload_files")
def upload_files():
    
    file_type = request.form['file_type']
    selected_option = request.form['selected_option']
    algorithm = request.form['algorithm']
  
    final_results, final_results_recommendations  = run_kmeans(selected_option,file_type, algorithm)
    final_results_pd = final_results.toPandas()
    final_results_reco_pd = final_results_recommendations.toPandas()
    
    # Επιστρέφουμε JSON με 2 keys
    return jsonify({
        "results": final_results_pd.to_dict(orient="records"),
        "recommendations": final_results_reco_pd.to_dict(orient="records")
    })


if __name__ == '__main__':
    app.run(debug=True)