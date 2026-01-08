
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



def run_kmeans( selected_option,file_type):
        spark = SparkSession.builder \
        .appName(f"Benchmark_Cores_{CORES}") \
        .master(f"local[{CORES}]") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
S
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
    

    #Add weightCol to feats_all
    feats_all = feats_all.withColumn(
        "weightCol",
        F.when(F.col("topic_merged_norm") == selected_option, F.lit(3)).otherwise(F.lit(1))
    )
    train_feats = feats_all.filter(F.col("topic_merged_norm") != "unknown") \
                        .select("scaled_features", "topic_merged_norm", "weightCol") \
                        .cache()



    # Evaluator: Silhouette με cosine 
    evaluator = ClusteringEvaluator(
        predictionCol="prediction",
        featuresCol="scaled_features",
        metricName="silhouette",
        distanceMeasure="cosine"
    )



    # Διάστημα τιμών k
    ks = list(range(10, 20))

    sil_scores = []
    models_by_k = {}

    best_sil = float("-inf")
    best_k = None
    best_model = None

    for k in ks:
        km = KMeans(featuresCol="scaled_features", k=k, seed=42, distanceMeasure="cosine", weightCol="weightCol")
        model = km.fit(train_feats)
        preds = model.transform(train_feats)
        sil = evaluator.evaluate(preds)
        sil_scores.append(sil)
        models_by_k[k] = model
        print(f"k={k:2d} → silhouette={sil:.6f}")
        
        # Κρατάμε το καλύτερο. Αν ισοπαλία ~0.01, προτιμάμε μικρότερο k (πιο απλό μοντέλο).
        if (sil > best_sil + 1e-6) or (abs(sil - best_sil) <= 0.01 and (best_k is None or k < best_k)):
            best_sil = sil
            best_k = k
            best_model = model
    
    #Αποθηκεύω σαν png 
    print(f"\nBest k = {best_k} with silhouette = {best_sil:.6f}")
    # Plot Silhouette vs k
    plt.figure()
    plt.plot(ks, sil_scores, marker='o')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette")
    plt.title("Silhouette vs k (cosine)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("silhouette_vs_k.png", dpi=150)
    plt.close()


    clusters_all = (
    best_model
        .transform(feats_all.select(
            "scaled_features", "topic_merged_norm", "id", "name", "surname",
            "researchField", "company_rank", "profile", "company_name",
            "industry", "city", "weightCol"
        ))
        .withColumnRenamed("prediction", "cluster")
        .withColumn(
            "identity",
            F.when(
                (F.col("company_rank") == -1) &
                (F.col("profile") == "unknown") &
                (F.col("company_name") == "unknown") &
                (F.col("industry") == "unknown"),
                F.lit("researcher")
            )
            .when(
                (F.col("id") == -1) &
                (F.col("name") == "unknown") &
                (F.col("surname") == "unknown") &
                (F.col("researchField") == "unknown"),
                F.lit("start-up")
            )
            .otherwise(F.lit("Unknown"))
        )
)

    #PCA σε 2 διαστάσεις 
    train_pred = best_model.transform(train_feats)
    pca = PCAml(k=2, inputCol="scaled_features", outputCol="pca2")
    pca_model = pca.fit(train_pred)
    train_2d = pca_model.transform(train_pred)

    # Μετατροπή vector -> array για indexing
    train_2d = train_2d.withColumn("pca2_arr", vector_to_array("pca2"))

    pdf = (train_2d
        .select(F.col("pca2_arr")[0].alias("pc1"),
                F.col("pca2_arr")[1].alias("pc2"),
                F.col("prediction").alias("cluster"))
        .sample(False, 0.2, seed=42)
        .toPandas())

    # Scatter - Αποθηκευω σαν png 
    plt.figure()
    plt.scatter(pdf["pc1"], pdf["pc2"], c=pdf["cluster"])
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"KMeans (k={best_k}) — PCA(2D)")
    plt.colorbar(label="Cluster")
    plt.tight_layout()
    plt.savefig("pca_clusters.png", dpi=150)
    plt.close()

    #αποθήκευσε και εμφάνισε μόνο τα αποτελέσματα με weightCol=3 την μεγαλύτερη προτεραιότητα 
    top_results = clusters_all.filter(clusters_all["weightCol"] == 3)
    #top_results.show()

 
    
    #επιλέγει και βρίσκει το cluster που βρίσκεται το topic_merged_norm "construction".Επέλεξε το cluster όπου το topic_merged_norm ισούται selected_option
    cluster_id=clusters_all.filter(clusters_all["topic_merged_norm"] == selected_option).select("cluster").first()["cluster"]
    #εμφάνιση όλων των αποτελεσμάτων που ανήκουν σε αυτό το cluster
    cluster_results = clusters_all.filter(clusters_all["cluster"] == cluster_id)
    #cluster_results.show()

    other_results = cluster_results.filter(cluster_results["weightCol"] != 3)
    final_results = top_results.union(other_results)
    final_results = final_results.drop("scaled_features")

    # Φιλτράρισμα βάσει file_type 
    if file_type == "start-up":
        final_results = final_results.filter(F.col("identity") == "researcher")

    elif file_type == "researcher":
        final_results = final_results.filter(F.col("identity") == "start-up")

    # Οι πιο σημαντικές εγγραφές (weightCol = 3)
    final_results_main = final_results.filter(F.col("weightCol") == 3)

    # Οι υπόλοιπες εγγραφές : Μπαίνουν στις "προτάσεις"
    final_results_recommendations = final_results.filter(F.col("weightCol") != 3)
    final_results.toPandas().to_csv('final_results.csv', index=False)
    final_results_recommendations.toPandas().to_csv('final_results_recommendations.csv', index=False)

    t0=time.time()
    print(final_results.count())
    t1=time.time()
    print(t1-t0)
    return final_results_main, final_results_recommendations

