from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, to_date, expr
from pyspark.sql.types import StructType, StructField, StringType, LongType, IntegerType, DoubleType
import logging

# --- Cấu hình Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Cấu hình ---
KAFKA_BROKER = "kafka-h2dn:29092"
KAFKA_TOPIC = "gold-price-data"
ES_NODES = "elasticsearch-h2dn"
ES_PORT = "9200"
ES_INDEX = "gold_prices_prod"
CHECKPOINT_LOCATION = "hdfs://namenode-h2dn:9000/user/spark/checkpoints/gold_stream_prod_checkpoint"

# Khởi tạo Spark Session với cấu hình HDFS
spark = SparkSession.builder \
    .appName("GoldPriceKafkaToElasticsearch_Prod") \
    .config("spark.streaming.stopGracefullyOnShutdown", "true") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.2.1,org.elasticsearch:elasticsearch-spark-30_2.12:8.9.0") \
    .config("spark.sql.streaming.checkpointLocation", CHECKPOINT_LOCATION) \
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode-h2dn:9000") \
    .config("spark.hadoop.dfs.client.use.datanode.hostname", "true") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
logger.info("SparkSession created.")

# Kiểm tra và tạo thư mục checkpoint trên HDFS
def ensure_hdfs_checkpoint(spark, checkpoint_path):
    try:
        fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
        path = spark._jvm.org.apache.hadoop.fs.Path(checkpoint_path)
        if not fs.exists(path):
            fs.mkdirs(path)
            logger.info(f"Created checkpoint directory: {checkpoint_path}")
        else:
            logger.info(f"Checkpoint directory already exists: {checkpoint_path}")
    except Exception as e:
        logger.error(f"ERROR ensuring HDFS checkpoint directory: {e}")
        raise

ensure_hdfs_checkpoint(spark, CHECKPOINT_LOCATION)

# Định nghĩa schema của dữ liệu JSON trong Kafka
schema = StructType([
    StructField("crawl_timestamp", LongType(), True),
    StructField("price_date", StringType(), True),
    StructField("gold_type", StringType(), True),
    StructField("buy_price", IntegerType(), True),
    StructField("sell_price", IntegerType(), True),
    StructField("source", StringType(), True)
])
logger.info("Schema defined.")

# Đọc dữ liệu từ Kafka
try:
    kafka_df = spark \
        .readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BROKER) \
        .option("subscribe", KAFKA_TOPIC) \
        .option("startingOffsets", "earliest") \
        .load()
    logger.info("Kafka stream loaded.")
except Exception as e:
    logger.error(f"ERROR loading Kafka stream: {e}")
    spark.stop()
    exit()

# Chuyển đổi dữ liệu từ Kafka (value là JSON) thành các cột
value_df = kafka_df.selectExpr("CAST(value AS STRING) as json_value")
parsed_df = value_df.select(from_json(col("json_value"), schema).alias("data")).select("data.*")
logger.info("JSON parsed.")

# Xử lý và Chuẩn hóa Kiểu dữ liệu
transformed_df = parsed_df \
    .withColumn("price_date_dt", to_date(col("price_date"), "yyyy-MM-dd")) \
    .withColumn("@timestamp", expr("CAST(crawl_timestamp / 1000 AS TIMESTAMP)")) \
    .withColumn("buy_price_dbl", col("buy_price").cast(DoubleType())) \
    .withColumn("sell_price_dbl", col("sell_price").cast(DoubleType()))
    

logger.info("Data transformed:")
transformed_df.printSchema()

# Ghi dữ liệu vào Elasticsearch
try:
    query = transformed_df \
        .writeStream \
        .format("org.elasticsearch.spark.sql") \
        .option("es.nodes", ES_NODES) \
        .option("es.port", ES_PORT) \
        .option("es.resource", ES_INDEX) \
        .option("es.mapping.id", "crawl_timestamp") \
        .option("es.write.operation", "index") \
        .option("es.nodes.wan.only", "true") \
        .option("es.index.auto.create", "true") \
        .option("checkpointLocation", CHECKPOINT_LOCATION) \
        .outputMode("append") \
        .trigger(processingTime='1 minute') \
        .start()

    logger.info(f"Writing stream to Elasticsearch index '{ES_INDEX}'...")
    query.awaitTermination()
except Exception as e:
    logger.error(f"ERROR writing to Elasticsearch: {e}")
    spark.stop()
    exit()
finally:
    spark.stop()
    logger.info("Spark Streaming job finished.")