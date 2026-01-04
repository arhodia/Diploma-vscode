
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



def run_kmeans( selected_option,file_type):
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
            .load("C:/Users/arhod/Desktop/Diploma-vscode/synthetic_files/synthetic_researchers_20000_inc5000dist.csv")


    # Cast σε συμβατούς τύπους για κοινό schema 
    researchers_df = researchers_df.withColumn("id", F.col("id").cast("int"))
    startups_df    = startups_df.withColumn("rank", F.col("rank").cast("int"))

    # Ορίζουμε το ΚΟΙΝΟ σετ στηλών που θέλουμε στο ενιαίο dataset και από τα 2 csv
    cols = ["id", "name", "surname", "researchField", "company_rank", "profile", "company_name", "industry", "city"]

    # Ετοιμάζουμε το DF των researchers: προσθέτουμε τις «εταιρικές» στήλες ως NULL
    r2 = (researchers_df
        .withColumn("company_rank", F.lit(None).cast("int"))
        .withColumn("profile", F.lit(None).cast("string"))
        .withColumn("company_name", F.lit(None).cast("string"))
        .withColumn("industry", F.lit(None).cast("string"))
        .withColumn("city", F.lit(None).cast("string"))
        .select(*cols))

    # Ετοιμάζουμε το DF των startups: μετονομάζουμε & προσθέτουμε τις «researcher» στήλες ως NULL
    s2 = (startups_df
        .withColumnRenamed("rank", "company_rank")
        .withColumnRenamed("name", "company_name")
        .select("company_rank", "profile", "company_name", "industry", "city")
        .withColumn("id", F.lit(None).cast("int"))
        .withColumn("name", F.lit(None).cast("string"))
        .withColumn("surname", F.lit(None).cast("string"))
        .withColumn("researchField", F.lit(None).cast("string"))
        .select(*cols))


    # Τελικό ενιαίο DataFrame με unionByName
    combined = r2.unionByName(s2, allowMissingColumns=True)
    combined = combined.dropDuplicates()
    combined = combined.distinct()
 
    missing_tokens = [
        "", " ", "  ",  # empty / whitespace
        "nan", "NaN", "NAN",
        "null", "NULL",
        "none", "None", "NONE",
        "NaT",
        "N/A", "n/a", "NA",
        "nand", "NAND"
    ]

    def summarize_missing(df, missing_tokens):
        tokens = [t.strip().lower() for t in missing_tokens if t is not None]

        schema = {f.name: f.dataType for f in df.schema.fields}
        string_cols = [n for n, dt in schema.items() if isinstance(dt, T.StringType)]
        float_like  = [n for n, dt in schema.items() if isinstance(dt, (T.FloatType, T.DoubleType))]

        def is_missing_str(colname):
            c = F.trim(F.col(colname))
            return F.col(colname).isNull() | (c == "") | F.lower(c).isin(tokens)

        # aggregations ανά στήλη
        agg_exprs = []
        for c in df.columns:
            if c in string_cols:
                cond = is_missing_str(c)
            elif c in float_like:
                cond = F.col(c).isNull() | F.isnan(F.col(c))
            else:
                cond = F.col(c).isNull()
            agg_exprs.append(F.sum(F.when(cond, 1).otherwise(0)).alias(c))

        counts_row = df.agg(*agg_exprs).collect()[0].asDict()

        summary_min = (
            df.sparkSession.createDataFrame([(k, int(v)) for k, v in counts_row.items()],
                                            ["column", "missing_count"])
            .orderBy(F.desc("missing_count"))
        )

        missing_cols = [r["column"]
                        for r in summary_min.filter(F.col("missing_count") > 0)
                                            .select("column").collect()]
        return summary_min, missing_cols

    summary_min,missing_cols = summarize_missing(combined, missing_tokens)

    def replace_missing_values(df, missing_tokens, cols=None, unknown="unknown",int_fill=-1,  long_fill=-1):
        
        # κανονικοποίηση tokens
        tokens = [t.strip().lower() for t in missing_tokens if t is not None]
        
        schema = {f.name: f.dataType for f in df.schema.fields}
        # αν δεν δοθούν ρητά cols, γέμισε όλες τις string στήλες
        if cols is None:
            cols = [f.name for f in df.schema.fields if isinstance(f.dataType, T.StringType)]
        
        def is_missing_str(cname):
            c = F.trim(F.col(cname))
            return F.col(cname).isNull() | (c == "") | F.lower(c).isin(tokens)

        out = df
        # Για κάθε επιλεγμένη στήλη, αντικατάστησε ανάλογα με τον τύπο της
        for c in cols:
            dt = schema.get(c)

            if isinstance(dt, T.StringType):
                # STRING: αντικατάσταση όλων των "κενών" με 'unknown'
                out = out.withColumn(
                    c,
                    F.when(is_missing_str(c), F.lit(unknown)).otherwise(F.col(c))
                )

            elif isinstance(dt, T.IntegerType):
                # INT: αντικατάσταση ΜΟΝΟ των NULL με int_fill (π.χ. -1)
                out = out.withColumn(
                    c,
                    F.when(F.col(c).isNull(), F.lit(int_fill).cast("int")).otherwise(F.col(c))
                )

            elif isinstance(dt, T.LongType):
                # LONG: αντικατάσταση ΜΟΝΟ των NULL με long_fill (π.χ. -1)
                out = out.withColumn(
                    c,
                    F.when(F.col(c).isNull(), F.lit(long_fill).cast("long")).otherwise(F.col(c))
                )

            else:
                # Άλλοι τύποι (Float/Double/Date/Timestamp/Boolean/Array/Struct κ.λπ.) δεν πειράζονται εδώ.
                pass

        return out

    clean_combined = replace_missing_values(combined,missing_tokens,cols=missing_cols,unknown="unknown",int_fill=-1,long_fill=-1)

    rf_norm  = F.lower(F.trim(F.col("researchField")))
    ind_norm = F.lower(F.trim(F.col("industry")))
    # Αναθέτει τιμή από τη στήλη "researchField" αν η στήλη rf_norm είναι μη κενή, μη "unknown".
    # Αν όχι, αναθέτει τιμή από "industry" όταν ind_norm πληροί το ίδιο κριτήριο. 
    # Στην αντίθετη περίπτωση βάζει "unknown".
    df_topics = (
        clean_combined
        .withColumn(
            "topic_merged",
            F.when(rf_norm.isNotNull() & (rf_norm != "") & (rf_norm != "unknown"), F.col("researchField"))
            .when(ind_norm.isNotNull() & (ind_norm != "") & (ind_norm != "unknown"), F.col("industry"))
            .otherwise(F.lit("unknown"))
        )
        # Καθαρίζει και κανονικοποιεί τη στήλη "topic_merged" σε πεζά, αφαιρώντας μη αλφαριθμητικούς χαρακτήρες.
        .withColumn(
            "topic_merged_norm",
            F.trim(F.regexp_replace(F.lower(F.col("topic_merged")), r"[^a-z0-9\s]+", " "))
        )
    )

    train_df = df_topics.filter(F.col("topic_merged_norm") != "unknown")

    #Ακολουθούν 6 εντολές 
    tok = RegexTokenizer(inputCol="topic_merged_norm", outputCol="tokens", pattern="\\W+")
    rmv = StopWordsRemover(inputCol="tokens", outputCol="tokens_no_sw")
    w2v = MLWord2Vec(
        inputCol="tokens_no_sw", outputCol="w2v",
        vectorSize=400, minCount=1, seed=42, maxIter=20, windowSize=5
    )
    scaler = StandardScaler(inputCol="w2v", outputCol="scaled_features", withMean=True, withStd=True)
    text_pipe = Pipeline(stages=[tok, rmv, w2v, scaler])
    text_model = text_pipe.fit(train_df)
    feats_all   = text_model.transform(df_topics)
    

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

