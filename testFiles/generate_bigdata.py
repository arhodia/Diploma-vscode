import os, math, glob, shutil, datetime
from pyspark.sql import SparkSession, functions as F, types as T

IN_PATH  = r"C:\\Users\\arhod\\Desktop\\Diploma-vscode\\indian_faculty_dataset.csv"
OUT_PATH = r"C:\\Users\\arhod\\Desktop\\Diploma-vscode\\indian_faculty_dataset_augmented_22MB.csv"

TARGET_MB = 22      # στόχος μεγέθους
DUP_FACTOR = 5     # περίπου 2MB * 5 ≈ 22MB
P_CHANGE = 0.5      # πιθανότητα να αλλάξει μια text στήλη
SEED = 42

spark = (SparkSession.builder
         .appName("AugmentNoCopies")
         .master("local[*]")
         .config("spark.hadoop.io.native.lib.available", "false")
         .getOrCreate())


# 1) Διάβασε
df = (spark.read.option("header", True).option("inferSchema", True).csv(IN_PATH))

# 2) Καθάρισε & κάνε casting στα 2 πεδία που είναι string με αριθμούς
df = (df
    .withColumn("Start Year",
                F.regexp_replace(F.col("Start Year"), r"[^0-9\-]", "").cast(T.IntegerType()))
    .withColumn("Years of Experience",
                F.floor(
                    F.regexp_replace(F.col("Years of Experience"), r"[^0-9\.\-]", "").cast(T.DoubleType())
                ).cast(T.IntegerType()))
)

# 3) Πολλαπλασίασε γραμμές (replication)
dup_df = spark.range(DUP_FACTOR).toDF("dup_idx")
aug = df.crossJoin(dup_df)

# 4) TEXT augmentation χωρίς UDFs:
# φτιάχνουμε literal arrays με distinct τιμές και επιλέγουμε ΜΙΑ ΑΛΛΗ (όχι την ίδια)
def literal_array_from_distinct(df, colname):
    vals = [r[0] for r in df.select(colname).where(F.col(colname).isNotNull()).distinct().collect()]
    return F.array(*[F.lit(v) for v in vals]) if vals else F.array()

def swap_expr(colname, arr_lit):
    cur = F.col(colname)
    # αφαίρεσε την τρέχουσα τιμή από το σύνολο, ανακάτεψε, πάρε 1ο στοιχείο
    others = F.array_except(arr_lit, F.array(cur))  # θέλει Spark array_except/shuffle/element_at
    pick   = F.when(F.size(others) > 0, F.element_at(F.shuffle(others), 1)).otherwise(cur)
    return F.when(F.col("dup_idx")==0, cur) \
            .otherwise(F.when(cur.isNull(), cur)  # αν είναι null, άστο null
                        .otherwise(F.when(F.rand(SEED) >= F.lit(P_CHANGE), cur).otherwise(pick)))

text_cols = ["Name","Position","Department","University","Location","Expertise",
             "Qualification","Highest Qualification"]
arrs = {c: literal_array_from_distinct(df, c) for c in text_cols}
for c in text_cols:
    aug = aug.withColumn(c, swap_expr(c, arrs[c]))
# (Το παραπάνω βασίζεται στα Spark built-ins: array_except/shuffle/element_at). :contentReference[oaicite:2]{index=2}

# 5) NUMERIC augmentation: Start Year / Years of Experience με μικρές, λογικές αλλαγές
current_year = datetime.datetime.now().year
year_delta = (F.floor(F.rand(SEED) * 5) - F.lit(2)).cast("int")   # -2..+2
yoe_jitter = (F.floor(F.rand(SEED+1) * 3) - F.lit(1)).cast("int") # -1..+1

aug = aug.withColumn(
    "Start Year",
    F.when(F.col("dup_idx")==0, F.col("Start Year"))
     .otherwise(F.when(F.col("Start Year").isNull(), None)
                 .otherwise(F.col("Start Year") + year_delta))
)

aug = aug.withColumn(
    "Years of Experience",
    F.when(F.col("dup_idx")==0, F.col("Years of Experience"))
     .otherwise(
        F.when(F.col("Start Year").isNull(),
               F.when(F.col("Years of Experience").isNull(), None)
                .otherwise(F.greatest(F.lit(0), F.col("Years of Experience") + yoe_jitter))
        ).otherwise(
            F.greatest(F.lit(0),
                       (F.lit(current_year) - F.col("Start Year")) + yoe_jitter)
        )
     ).cast(T.IntegerType())
)

# 6) Νέο μοναδικό ID (για να μην έχεις ίδιους αναγνωριστές μετά το replication)
aug = aug.withColumn("synthetic_id", F.monotonically_increasing_id())  # unique/monotonic id :contentReference[oaicite:3]{index=3}

# 7) Ρίξε τη βοηθητική στήλη
aug = aug.drop("dup_idx")


# 8) Γράψε ΕΝΑ CSV (χωρίς συμπίεση)
tmpdir = OUT_PATH + "_tmpdir"
(aug.coalesce(1)
    .write.mode("overwrite")
    .option("header", True)
    .option("compression","none")
    .csv(tmpdir))
part = [p for p in os.listdir(tmpdir) if p.startswith("part-") and p.endswith(".csv")][0]
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
shutil.move(os.path.join(tmpdir, part), OUT_PATH)
shutil.rmtree(tmpdir)

print("OK →", OUT_PATH)
spark.stop()