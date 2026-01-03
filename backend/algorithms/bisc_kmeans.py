
from pyspark.ml.functions import vector_to_array
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover
from pyspark.ml.feature import Word2Vec as MLWord2Vec, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql import types as T
import os
from pyspark.sql import functions as F
from pyspark.ml.feature import PCA as PCAml
import time
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use("Agg") #Χρησιμοποιω το Agg backend για αποθήκευση εικόνων χωρίς GUI

os.environ["PYSPARK_PYTHON"] = "C:\\Users\\arhod\\AppData\\Local\\Programs\\Python\\Python310\\python.exe"
os.environ["PYSPARK_DRIVER_PYTHON"] = "C:\\Users\\arhod\\AppData\\Local\\Programs\\Python\\Python310\\python.exe"


spark = (SparkSession.builder
                .appName("Researchers+Companies: Unified KMeans")
                .master("local[2]").getOrCreate())
    
spark.sparkContext.setLogLevel("WARN")
    
print(spark.sparkContext.master)
print(spark.sparkContext.defaultParallelism) 
    
startups_df = spark.read.format("csv") \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .load("C:/Users/arhod/Desktop/Diploma-vscode/INC 5000 Companies 2019.csv")

researchers_df = spark.read.format("csv") \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .load("C:/Users/arhod/Desktop/Diploma-vscode/synthetic_researchers_20000_inc5000dist.csv")
    
startups_df.show(10)
researchers_df.show(10)

