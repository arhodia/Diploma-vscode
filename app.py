import numpy as np
import pandas as pd
from flask import Flask, jsonify
from flask_cors import CORS
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
print("Second CSV data:")
print(dataINC5000)
# To display the data in a React UI, you need to expose it via an API.
# Here is a simple Flask API to serve the CSV data as JSON.
# >>> ΚΛΕΙΔΙ: Μετατροπή NaN -> None (ώστε να πάει ως null στο JSON)
dataINC5000 = dataINC5000.replace({np.nan: None})


@app.get('/api/inc5000')
def get_inc5000():
    dataINC = dataINC5000.head(20).to_dict(orient='records')
    return jsonify(dataINC)

if __name__ == '__main__':
    app.run(debug=True)