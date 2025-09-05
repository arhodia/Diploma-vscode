import csv
import pandas as pd
from flask import Flask, jsonify
# Read the first CSV file as a DataFrame
dataindian_faculty_dataset = pd.read_csv('C:\\Users\\arhod\\Desktop\\Diploma-vscode\\indian_faculty_dataset.csv')
dataINC5000 = pd.read_csv('C:\\Users\\arhod\\Desktop\\Diploma-vscode\\INC 5000 Companies 2019.csv')

# Count the size of the DataFrames in MB
size_indian_faculty_mb = dataindian_faculty_dataset.memory_usage(deep=True).sum() / (1024 * 1024)
size_inc5000_mb = dataINC5000.memory_usage(deep=True).sum() / (1024 * 1024)

print(f"First CSV data size: {size_indian_faculty_mb:.2f} MB")
print(f"Second CSV data size: {size_inc5000_mb:.2f} MB")

# Example: print the contents
print("First CSV data:")
print(dataindian_faculty_dataset)
print("Second CSV data:")
print(dataINC5000)
# To display the data in a React UI, you need to expose it via an API.
# Here is a simple Flask API to serve the CSV data as JSON.


app = Flask(__name__)

@app.route('/api/indian_faculty')
def get_indian_faculty():
    return dataindian_faculty_dataset.to_json(orient='records')

@app.route('/api/inc5000')
def get_inc5000():
    return dataINC5000.to_json(orient='records')

if __name__ == '__main__':
    app.run(debug=True)