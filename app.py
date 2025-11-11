import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
import os
from backend.test import cluster_topics_from_files
app = Flask(__name__)
CORS(app)
# Read the first CSV file as a DataFrame
dataINC5000 = pd.read_csv('C:\\Users\\arhod\\Desktop\\Diploma-vscode\\INC 5000 Companies 2019.csv')

# Count the size of the DataFrames in MB
size_inc5000_mb = dataINC5000.memory_usage(deep=True).sum() / (1024 * 1024)

'''
print(f"First CSV data size: {size_indian_faculty_mb:.2f} MB")
print(f"Second CSV data size: {size_inc5000_mb:.2f} MB")
'''

DEFAULT_RESEARCH = "C:\\Users\\arhod\\Desktop\\Diploma-vscode\\synthetic_researchers_5000_base.csv"
DEFAULT_STARTUPS = "C:\\Users\\arhod\\Desktop\\Diploma-vscode\\INC 5000 Companies 2019.csv"


dataINC5000 = dataINC5000.replace({np.nan: None})

@app.get("/")
def get_clusters():
    clusters_all = cluster_topics_from_files(DEFAULT_STARTUPS, DEFAULT_RESEARCH)
    clusters_all_pd = clusters_all.toPandas()
    # Παράδειγμα: ομαδοποίηση και επιλογή top topics ανά cluster
    result = (
        clusters_all_pd
            .groupby("cluster")["topic_merged_norm"]
            .value_counts()
            .groupby(level=0)
            .head(15)
            .reset_index(name='count')
            .to_dict(orient="records")
    )
    return jsonify(result)



@app.post("/api/upload_files")
def upload_files():
    # Τα αρχεία που στέλνει ο client (π.χ. με fetch/axios σε React)
    research_file = request.files['research_file']
    startup_file  = request.files['startup_file']
    """
    # Αποθηκεύεις προσωρινά ή τα περνάς ως file-like io
    research_path = "/tmp/research.csv"
    startup_path = "/tmp/startup.csv"
    research_file.save(research_path)
    startup_file.save(startup_path)
    """
    # Κλήση της Spark function με paths των uploaded files
    clusters_all = cluster_topics_from_files(startup_file, research_file)
    clusters_all_pd = clusters_all.toPandas()
    result = clusters_all_pd[['cluster','topic_merged_norm']].head(10).to_dict(orient='records')
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)