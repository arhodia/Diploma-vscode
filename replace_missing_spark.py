""" # Step 1: Data Preprocessing with PySpark
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col,lit,concat_ws
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql.functions import col
import pandas as pd
from pyspark.sql.functions import col, mean as _mean, count as _count, when
from pyspark.sql.types import NumericType
from pyspark.sql.functions import isnan
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import col, when, lit, isnan
from pyspark.sql.types import NumericType
import datetime

# Initialize Spark session
spark = SparkSession.builder.appName("ResearcherStartupMatching").getOrCreate()

# Load datasets
# researchers_df = spark.read.option("header", "true").csv("C:/Users/arhod/Desktop/Diploma-vscode/indian_faculty_dataset.csv")
# Count the number of rows for each dataset
researchers_count = spark.read.option("header", "true").csv("C:/Users/arhod/Desktop/Diploma-vscode/indian_faculty_dataset.csv").count()
startups_count = spark.read.option("header", "true").csv("C:/Users/arhod/Desktop/Diploma-vscode/INC 5000 Companies 2019.csv").count()


print(f"Number of rows in researchers dataset: {researchers_count}")
print(f"Number of rows in startups dataset: {startups_count}") 


startups_df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load("C:/Users/arhod/Desktop/Diploma-vscode/INC 5000 Companies 2019.csv")

researchers_df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load("C:/Users/arhod/Desktop/Diploma-vscode/indian_faculty_dataset.csv")

# count how many nan, null, and ' ' values are in the dataframe
def count_null_nan(df):
    null_nan_counts = {}
    for column in df.columns:
        count = df.filter(
            col(column).isNull() | (col(column) != col(column)) | (col(column) == ' ')
        ).count()
        null_nan_counts[column] = count
    return null_nan_counts

# call the count_null_nan function and calculate null and nan values
null_nan_counts_startups = count_null_nan(startups_df)
null_nan_counts_researchers = count_null_nan(researchers_df)
print("Startups null/nan counts:", null_nan_counts_startups)
print("Researchers null/nan counts:", null_nan_counts_researchers)

print("Researchers dataframe columns are:")
researchers_df.printSchema()
print("Startups dataframe columns are:")
startups_df.printSchema()


# Μετατροπή των δύο στηλών από string (με μορφή 3.0, 4.0, κλπ.) σε integer
researchers_df_new = (
    researchers_df
    .withColumn("Start Year", col("Start Year").cast(IntegerType()))
    .withColumn("Years of Experience", col("Years of Experience").cast(IntegerType()))
)

# replace the nan and null values with a parameter mean
#def replace_missing_spark(df, columns, method="mean"):
#    for column in columns:
        # Check for null, NaN, or space (' ') values
#        null_count = df.filter(
#            col(column).isNull() | (col(column) != col(column)) | (col(column) == ' ')
#       ).count()
#        if null_count > 0:
#            if method == "mean":
#                mean_value = df.select(_mean(col(column))).first()[0]
#                if mean_value is not None:
#                    df = df.withColumn(
#                        column,
#                        when(
#                            col(column).isNull() | (col(column) != col(column)) | (col(column) == ' '),
#                           mean_value
#                        ).otherwise(col(column))
#                    )
#            elif method == "count":
#                count_value = df.count()
#                df = df.withColumn(
#                    column,
#                    when(
#                        col(column).isNull() | (col(column) != col(column)) | (col(column) == ' '),
#                        count_value
#                    ).otherwise(col(column))
#                )
    # Add more methods if needed
#    return df


def replace_missing_spark(df, columns, numeric_fill="max", string_fill="Unknown"):
    current_year = datetime.datetime.now().year  # π.χ. 2025
    
    for column in columns:
        if column == "Years of Experience":
            # Αν η στήλη έχει κενό → current_year - Start Year
            df = df.withColumn(
                column,
                when(
                    (col(column) == ' ') | col(column).isNull() | isnan(col(column)),
                    lit(current_year) - col("Start Year")
                ).otherwise(col(column))
            )
        
        elif isinstance(df.schema[column].dataType, NumericType):
            if numeric_fill == "max":
                max_value = df.select(col(column)).agg({column: "max"}).first()[0]
                if max_value is not None:
                    df = df.withColumn(
                        column,
                        when(
                            col(column).isNull() | isnan(col(column)) | (col(column) == ' '),
                            max_value
                        ).otherwise(col(column))
                    )
            else:
                df = df.fillna({column: numeric_fill})
        
        else:
            df = df.withColumn(
                column,
                when(
                    col(column).isNull() | (col(column) == '') | isnan(col(column)),
                    string_fill
                ).otherwise(col(column))
            )
    return df


# insert columns to replace 
columns_to_replace_startups = ["workers", "metro"]
startups_df = replace_missing_spark(startups_df, columns_to_replace_startups, numeric_fill="max", string_fill="Unknown")

columns_to_replace_researchers = ["Department", "Location", "Expertise", "Experience", "Qualification", "Honours and Awards", "Start Year", "Years of Experience"]
researchers_df_new = replace_missing_spark(researchers_df_new, columns_to_replace_researchers, numeric_fill="max", string_fill="Unknown")


# check if there is null values in the new dataframes
null_nan_counts_startups = count_null_nan(startups_df)
null_nan_counts_researchers = count_null_nan(researchers_df_new)
print("Startups null/nan counts:", null_nan_counts_startups)
print("Researchers null/nan counts:", null_nan_counts_researchers) 


# Print the row from researchers_df where 'vidwan-id' equals 56586
researchers_df_new.filter(col("vidwan-id") == 56586).show()

print("Researchers dataframe columns are:")
researchers_df_new.printSchema()  """
# replace_missing_spark.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit, isnan
from pyspark.sql.types import IntegerType, NumericType
import datetime

# Initialize Spark session 
spark = SparkSession.builder.appName("ResearcherStartupMatching").getOrCreate()

#Συνάρτηση που υπολογίζει τα null/nan values για όλες τις στήλες
def count_null_nan(df):
    null_nan_counts = {}
    for column in df.columns:
        count = df.filter(
            col(column).isNull() | (col(column) != col(column)) | (col(column) == ' ')
        ).count()
        null_nan_counts[column] = count
    return null_nan_counts

#Συνάρτηση που αντικαθιστά τα null/nan values ανάλογα με τον τύπο της στήλης
def replace_missing_spark(df, columns, numeric_fill="max", string_fill="Unknown"):
    current_year = datetime.datetime.now().year
    for column in columns:
        if column == "Years of Experience":
            df = df.withColumn(
                column,
                when(
                    (col(column) == ' ') | col(column).isNull() | isnan(col(column)),
                    lit(current_year) - col("Start Year")
                ).otherwise(col(column))
            )
        elif isinstance(df.schema[column].dataType, NumericType):
            if numeric_fill == "max":
                max_value = df.select(col(column)).agg({column: "max"}).first()[0]
                if max_value is not None:
                    df = df.withColumn(
                        column,
                        when(
                            col(column).isNull() | isnan(col(column)) | (col(column) == ' '),
                            max_value
                        ).otherwise(col(column))
                    )
            else:
                df = df.fillna({column: numeric_fill})
        else:
            df = df.withColumn(
                column,
                when(
                    col(column).isNull() | (col(column) == '') | isnan(col(column)),
                    string_fill
                ).otherwise(col(column))
            )
    return df


def load_and_clean_data():
    startups_df = spark.read.format("csv") \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .load("C:/Users/arhod/Desktop/Diploma-vscode/INC 5000 Companies 2019.csv")
    # Find the first row where 'workers' column is ' ' and get the entire row
    first_blank_workers_row = startups_df.filter(col("workers") == ' ').first()
    

    researchers_df = spark.read.format("csv") \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .load("C:/Users/arhod/Desktop/Diploma-vscode/indian_faculty_dataset.csv")

    null_nan_counts_researchers = count_null_nan(researchers_df)
    null_nan_counts_startups = count_null_nan(startups_df)
    researchers_df = (
        researchers_df
        .withColumn("Start Year", col("Start Year").cast(IntegerType()))
        .withColumn("Years of Experience", col("Years of Experience").cast(IntegerType()))
    )

    startups_df = replace_missing_spark(startups_df, ["workers", "metro"], numeric_fill="max", string_fill="Unknown")
    researchers_df = replace_missing_spark(researchers_df, 
        ["Department", "Location", "Expertise", "Experience", "Qualification", "Honours and Awards", "Start Year", "Years of Experience"], 
        numeric_fill="max", string_fill="Unknown"
    )

    return  first_blank_workers_row,null_nan_counts_startups,null_nan_counts_researchers,startups_df, researchers_df
    return null_nan_counts_researchers, startups_df, researchers_df

