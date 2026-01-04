from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, monotonically_increasing_id
import os

# Ρυθμίσεις περιβάλλοντος
os.environ["PYSPARK_PYTHON"] = "C:\\Users\\arhod\\AppData\\Local\\Programs\\Python\\Python310\\python.exe"
os.environ["PYSPARK_DRIVER_PYTHON"] = "C:\\Users\\arhod\\AppData\\Local\\Programs\\Python\\Python310\\python.exe"

output_path = "C:/Users/arhod/Desktop/Diploma-vscode/output_union_data"
CORES = 9 

spark = SparkSession.builder \
    .appName(f"Benchmark_Cores_{CORES}") \
    .master(f"local[{CORES}]") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# --- ΒΗΜΑ 1: Φόρτωση Αρχείων ---
startups_df = spark.read.format("csv") \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .load("C:/Users/arhod/Desktop/Diploma-vscode/INC 5000 Companies 2019.csv")

researchers_df = spark.read.format("csv") \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .load("C:/Users/arhod/Desktop/Diploma-vscode/synthetic_files/synthetic_researchers_20000_inc5000dist.csv")

# --- ΒΗΜΑ 2: Προετοιμασία Στηλών (Renaming) ---
# Μετονομάζουμε τις στήλες ώστε να είναι ξεκάθαρο ποια ανήκει πού
companies_clean = startups_df \
    .withColumnRenamed("name", "company_name") \
    .withColumnRenamed("id", "company_id") \
    .withColumnRenamed("industry", "company_industry") \
    .withColumn("source_type", lit("Company")) 

researchers_clean = researchers_df \
    .withColumnRenamed("name", "researcher_name") \
    .withColumnRenamed("id", "researcher_id") \
    .withColumnRenamed("researchfield", "researcher_field") \
    .withColumn("source_type", lit("Researcher")) 

# --- ΒΗΜΑ 3: Η Σωστή Ένωση (UnionByName) ---
# Το allowMissingColumns=True είναι το κλειδί. 
union_df = researchers_clean.unionByName(companies_clean, allowMissingColumns=True)
union_df_with_id = union_df.withColumn("id", monotonically_increasing_id())
other_columns = [c for c in union_df_with_id.columns if c != "id"]

join_df = union_df_with_id.select("global_id", *other_columns)
# --- Αποθήκευση ---
join_df.coalesce(1) \
    .write \
    .mode("overwrite") \
    .option("header", "true") \
    .csv(output_path)

print(f"Αριθμός ερευνητών: {researchers_clean.count()}")
print(f"Αριθμός εταιρειών: {companies_clean.count()}")
print(f"Συνολικός αριθμός γραμμών (Πρέπει να είναι το άθροισμα): {union_df.count()}")
print(f"Το αρχείο αποθηκεύτηκε επιτυχώς στο: {output_path}")
print(f"Spark Version: {spark.version}")
print(f"Spark UI URL: {spark.sparkContext.uiWebUrl}")
print(f"Spark Master: {spark.sparkContext.master}") 