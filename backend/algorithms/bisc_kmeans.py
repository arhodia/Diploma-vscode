
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
from pyspark.sql.functions import col, coalesce, lit
matplotlib.use("Agg") #Χρησιμοποιω το Agg backend για αποθήκευση εικόνων χωρίς GUI
import os
import sys
os.environ["PYSPARK_PYTHON"] = "C:\\Users\\arhod\\AppData\\Local\\Programs\\Python\\Python310\\python.exe"
os.environ["PYSPARK_DRIVER_PYTHON"] = "C:\\Users\\arhod\\AppData\\Local\\Programs\\Python\\Python310\\python.exe"

spark = (SparkSession.builder
                .appName("Researchers+Companies: Unified KMeans")
                .master("local[8]").getOrCreate())
    
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

# ---------------------------------------------------------
# ΒΗΜΑ 2: Μόνο Μετονομασία (Χωρίς καθαρισμό revenue)
# ---------------------------------------------------------
# Απλά μετονομάζουμε για να αποφύγουμε το conflict στα ονόματα
companies_clean = startups_df \
    .withColumnRenamed("name", "company_name") \
    .withColumnRenamed("id", "company_id") \
    .withColumnRenamed("industry", "company_industry") 

# ---------------------------------------------------------
# ΒΗΜΑ 3: Ένωση (Full Outer Join)
# ---------------------------------------------------------
joined_df = researchers_df.join(companies_clean, 
                                researchers_df.researchfield == companies_clean.company_industry, 
                                "outer")

# ---------------------------------------------------------
# ΒΗΜΑ 4: Διαχείριση Κενών στο πεδίο Ομαδοποίησης
# ---------------------------------------------------------
# Δημιουργούμε την κοινή στήλη κατηγορίας
joined_df = joined_df.withColumn("merged_category", coalesce(col("researchfield"), col("company_industry")))

# Γεμίζουμε τα κενά ΜΟΝΟ στην κατηγορία (τα αριθμητικά δεν μας νοιάζουν πλέον)
joined_df = joined_df.fillna("Unknown", subset=["merged_category"])

# ---------------------------------------------------------
# ΒΗΜΑ 5: Πολλαπλασιασμός Δεδομένων (Stress Test Preparation)
# ---------------------------------------------------------
print("\n--- ΕΝΑΡΞΗ ΠΟΛΛΑΠΛΑΣΙΑΣΜΟΥ ---")
initial_count = joined_df.count()
print(f"Αρχικό πλήθος εγγραφών: {initial_count}")

# Κάνουμε Union τον εαυτό του με τον εαυτό του πολλές φορές
# x2
joined_df = joined_df.union(joined_df)
# x4
joined_df = joined_df.union(joined_df)
# x8
joined_df = joined_df.union(joined_df)
# x16 (Αν το PC αντέχει, μπορείς να προσθέσεις κι άλλα)
joined_df = joined_df.union(joined_df)

# ---------------------------------------------------------
# ΒΗΜΑ 6: Επιβεβαίωση Spark & Εκτύπωση
# ---------------------------------------------------------
print("\n--- ΕΛΕΓΧΟΣ ΜΕΤΑ ΤΟΝ ΠΟΛΛΑΠΛΑΣΙΑΣΜΟ ---")

# 1. Έλεγχος Τύπου: Εδώ φαίνεται αν χρησιμοποιούμε Spark
# Αν γράψει: <class 'pyspark.sql.dataframe.DataFrame'> τότε ΕΙΝΑΙ Spark.
# Αν έγραφε: <class 'pandas.core.frame.DataFrame'> τότε θα ήταν Pandas.
print(f"Τύπος DataFrame: {type(joined_df)}")

# 2. Εκτύπωση των 10 πρώτων σειρών
# Το .show() τρέχει στον Driver αλλά τραβάει δεδομένα από τους Workers
print("Δείγμα 10 εγγραφών:")
joined_df.select("name", "surname", "merged_category", "company_name").show(10, truncate=False)

# 3. Τελικό Πλήθος
# Το count() είναι "Action". Εδώ το Spark θα δουλέψει πραγματικά για να μετρήσει.
final_count = joined_df.count()
print(f"Τελικό πλήθος εγγραφών για το πείραμα: {final_count}")

# Σύγκριση
print(f"Τα δεδομένα αυξήθηκαν κατά {final_count / initial_count} φορές.")