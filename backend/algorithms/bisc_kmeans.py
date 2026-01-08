import time
import numpy as np
import os
from pyspark.sql.functions import  col
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, monotonically_increasing_id, coalesce, col
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, Word2Vec
from pyspark.ml.clustering import BisectingKMeans
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors
import matplotlib.pyplot as plt
from pyspark.sql import functions as F
from pyspark.ml.functions import vector_to_array
from pyspark.ml.linalg import Vectors

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

#Φόρτωση Αρχείων
startups_df = spark.read.format("csv") \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .load("C:/Users/arhod/Desktop/Diploma-vscode/INC 5000 Companies 2019.csv")

researchers_df = spark.read.format("csv") \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .load("C:/Users/arhod/Desktop/Diploma-vscode/synthetic_files/synthetic_researchers_20000_inc5000dist.csv")

#Προετοιμασία Στηλών (Renaming) ---
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

#  Η Σωστή Ένωση (UnionByName)
# Το allowMissingColumns=True είναι το κλειδί. 
union_df = researchers_clean.unionByName(companies_clean, allowMissingColumns=True)
union_df_with_id = union_df.withColumn("id", monotonically_increasing_id())
other_columns = [c for c in union_df_with_id.columns if c != "id"]

join_df = union_df_with_id.select("id", *other_columns)



#Δημιουργία κοινής στήλης χαρακτηριστικών researcher_field + company_industry
join_df = join_df.withColumn("industry", coalesce(col("researcher_field"), col("company_industry")))

# Αποθήκευση 
join_df.coalesce(1) \
    .write \
    .mode("overwrite") \
    .option("header", "true") \
    .csv(output_path)


tokenizer = RegexTokenizer(inputCol="industry", outputCol="words", pattern="\\W")
stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
word2vec = Word2Vec(inputCol="filtered_words", outputCol="features", vectorSize=50, minCount=0,maxIter=50, windowSize=5,seed=42)   
# Φτιάχνουμε Pipeline ΜΟΝΟ για τα features
feature_pipeline = Pipeline(stages=[tokenizer, stopwords_remover, word2vec])
feature_model = feature_pipeline.fit(join_df)
feature_df = feature_model.transform(join_df)

print(f"Spark Version: {spark.version}")
#  SOS: CACHE ΕΔΩ *** # Κρατάμε τα features στη μνήμη για να μην τα υπολογίζει ξανά στο Loop του Elbow
feature_df.cache()
#print(f"Data cached. Total rows: {feature_df.count()}")
k_range = range(2, 21) 
costs = []

print("Ξεκινάει ο υπολογισμός του Elbow Method...")

for k in k_range:
    # Ορισμός του BisectingKMeans
    bkm = BisectingKMeans(k=k, seed=42, featuresCol="features")
    
    # Εκπαίδευση μοντέλου
    model = bkm.fit(feature_df)

    try:
        cost = model.summary.trainingCost
    except AttributeError:
        # Fallback για παλαιότερες εκδόσεις Spark ή αν το summary δεν είναι διαθέσιμο
        cost = model.computeCost(feature_df)

    costs.append(cost)
    print(f"k={k} | Cost={cost:.2f} ")

#Εκτελώ τον bisecting kmeans με κ=9


optimal_k = 9
#φτιάχνει το αντικείμενο του Bisecting kmeans με συγκερκριμένα χαρακτηριστικά από τις υπερπαραμέτρους του bisecting kmeans
bkm = BisectingKMeans(k=optimal_k, seed=42, featuresCol="features", predictionCol="prediction")
#ξεκινά τη διαδικασία εκμάθησης πάνω στο feature_df 
final_model = bkm.fit(feature_df) 
#Χρησιμοποιεί το εκπαιδευμένο μοντέλο final_model και με βάση αυτό αναθέτει τιμές στην στήλη prediction για κάθε γραμμή του feature_df
#final_prediction = ολες οι στήλες του feature_df + η στήλη prediction με την τιμή του cluster
finaldf_prediction = final_model.transform(feature_df)


#Υπολογισμος similarity score
features_col = final_model.getFeaturesCol()

# 2) Cluster centers -> DataFrame (prediction -> center vector)
centers = final_model.clusterCenters()
centers_df = spark.createDataFrame(
    [(i, Vectors.dense(c)) for i, c in enumerate(centers)],
    ["prediction", "center"]
)

# 3) Join για να πάρει κάθε row το κέντρο του cluster του
df = finaldf_prediction.join(F.broadcast(centers_df), on="prediction", how="left")

# 4) Υπολογισμός Euclidean distance χωρίς UDF
df = (
    df
    .withColumn("features_arr", vector_to_array(F.col(features_col)))
    .withColumn("center_arr",   vector_to_array(F.col("center")))
    .withColumn(
        "distance",
        F.expr("""
            sqrt(
              aggregate(
                arrays_zip(features_arr, center_arr),
                0D,
                (acc, x) -> acc + pow(x.features_arr - x.center_arr, 2D)
              )
            )
        """)
    )
)


print(df.columns)
# Δες μερικά αποτελέσματα
df.select("prediction","researcher_id","researcher_name","surname","researcher_field","source_type", "rank","profile","company_name","url","distance").show(2, truncate=False)

"""
#τα αποτελεσματα του αλγοριθμου Bisecting KMeans για το συγκεκριμενο μοντέλο 
print(final_model.hasSummary)
print(final_model.summary.k)
print(final_model.summary.cluster)
print(final_model.distanceMeasure)
print(final_model.summary.trainingCost)
(print(final_model.getDistanceMeasure()))
centers = model.clusterCenters()
print(centers)

"""

"""
#Δημιουργία pca plot για αναπαράσταση αποτελεσμάτων
# --- Βήμα 1: Υπολογισμός PCA με Spark (όχι sklearn) ---
print("Εκτέλεση PCA μέσα στο Spark...")

# Ρυθμίζουμε το Spark PCA να μειώσει τις διαστάσεις από 50 σε 2
spark_pca = PCA(k=2, inputCol="features", outputCol="pca_features")

# Εκπαιδεύουμε το μοντέλο PCA πάνω στα δεδομένα μας
pca_model = spark_pca.fit(finaldf_prediction)

# Εφαρμόζουμε το PCA στα δεδομένα (Data Points)
pca_result_df = pca_model.transform(finaldf_prediction)

# --- Βήμα 2: Εφαρμογή του ίδιου PCA στα Κέντρα των Clusters ---
# Πρέπει να μετατρέψουμε τα κέντρα σε Spark DataFrame για να περάσουν από το ίδιο PCA μοντέλο

centers = final_model.clusterCenters()
# Μετατροπή των κέντρων (που είναι arrays) σε μορφή που καταλαβαίνει το Spark
centers_data = [(Vectors.dense(c),) for c in centers]
centers_df = spark.createDataFrame(centers_data, ["features"])

# Εφαρμογή του μοντέλου PCA και στα κέντρα
centers_pca_df = pca_model.transform(centers_df)

# --- Βήμα 3: Συλλογή αποτελεσμάτων για το γράφημα ---
# Παίρνουμε ένα δείγμα (π.χ. 20%) για να μην "πουκάρει" η μνήμη του driver στο plotting
# Χρησιμοποιούμε .collect() που φέρνει τα δεδομένα ως λίστα από Rows

rows = pca_result_df.select("pca_features", "prediction").sample(False, 0.2, seed=42).collect()
center_rows = centers_pca_df.select("pca_features").collect()

# Προετοιμασία λιστών για το Matplotlib ---
# Εδώ διαβάζουμε τα Vectors του Spark και τα βάζουμε σε απλές λίστες Python (X και Y)

x_points = [row['pca_features'][0] for row in rows]
y_points = [row['pca_features'][1] for row in rows]
cluster_ids = [row['prediction'] for row in rows]

x_centers = [row['pca_features'][0] for row in center_rows]
y_centers = [row['pca_features'][1] for row in center_rows]

#Σχεδίαση (Matplotlib) 
plt.figure(figsize=(12, 8))

# Σημεία (Data Points)
scatter = plt.scatter(x_points, y_points, c=cluster_ids, cmap='tab10', alpha=0.6, s=10, label='Data Points')

# Κέντρα (Centroids)
plt.scatter(x_centers, y_centers, c='red', marker='X', s=200, edgecolor='black', label='Cluster Centers')

# Ετικέτες στα κέντρα
for i, (x, y) in enumerate(zip(x_centers, y_centers)):
    plt.text(x, y, str(i), fontsize=12, weight='bold', color='black')

plt.title(f'Bisecting K-Means Clusters (Spark PCA) - k={optimal_k}')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.colorbar(scatter, label='Cluster ID')
plt.grid(True, alpha=0.3)

plt.show()

"""

print("Η διαδικασία ολοκληρώθηκε.")
spark.stop()
# Εμφανίζουμε τα top αποτελέσματα για να δούμε τι περιέχει το καθένα
# Το show(100) θα δείξει αρκετές γραμμές ώστε να δεις τα πρώτα industries για κάθε cluster

"""
#Επιβεβαίωση Αποτελεσμάτων
print(f"Αριθμός ερευνητών: {researchers_clean.count()}")
print(f"Αριθμός εταιρειών: {companies_clean.count()}")
print(f"Συνολικός αριθμός γραμμών (Πρέπει να είναι το άθροισμα): {union_df.count()}")
print(f"Το αρχείο αποθηκεύτηκε επιτυχώς στο: {output_path}")
print(f"Spark UI URL: {spark.sparkContext.uiWebUrl}")
print(f"Spark Master: {spark.sparkContext.master}") 
#Υπολογισμός του κόστους για επιβεβαίωση
final_cost = final_model.summary.trainingCost
print(f"Τελικό Κόστος (WSSSE): {final_cost:.2f}")
print(final_model)
# Εμφάνιση αποτελεσμάτων
#predictions_df.select("researcher_id","researcher_name","surname","researcher_field","university","age","source_type","rank","profile","company_name","url","state","company_industry","workers","previous_workers","metro","city","industry", "prediction").show(10)
"""

