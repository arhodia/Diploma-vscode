# replace_missing_spark.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit, isnan
from pyspark.sql.types import IntegerType, NumericType
import datetime

# Initialize Spark session 
spark = SparkSession.builder.appName("ResearcherStartupMatching").master("local[*]").getOrCreate()

#Διαβάζω τo 1o αρχείο σε spark
startups_df = spark.read.format("csv") \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .load("C:/Users/arhod/Desktop/Diploma-vscode/INC 5000 Companies 2019.csv")


researchers_df = spark.read.format("csv") \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .load("C:/Users/arhod/Desktop/Diploma-vscode/indian_faculty_dataset.csv")




