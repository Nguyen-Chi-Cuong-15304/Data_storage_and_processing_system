from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, to_date, year, month, dayofmonth
from pyspark.sql.types import StructType, StructField, StringType, LongType, IntegerType
import logging

# --- Cấu hình Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Cấu hình ---
KAFKA_BROKER = "kafka-h2dn:29092"
KAFKA_TOPIC = "gold-price-data"
HDFS_RAW_DATA_PATH = "hdfs://namenode-h2dn:9000/user/data/gold_raw_parquet"
CHECKPOINT_LOCATION_HDFS = "hdfs://namenode-h2dn:9000/user/spark/checkpoints/kafka_to_hdfs_checkpoint"

def main():
    logger.info("Starting Spark Streaming job: Kafka to HDFS Raw Parquet")

    # Khởi tạo Spark Session với cấu hình HDFS
    spark = SparkSession \
        .builder \
        .appName("KafkaToHDFSRaw") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.2.1") \
        .config("spark.sql.streaming.checkpointLocation", CHECKPOINT_LOCATION_HDFS) \
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode-h2dn:9000") \
        .config("spark.hadoop.dfs.client.use.datanode.hostname", "true") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    logger.info("SparkSession created.")

    # Kiểm tra và tạo thư mục HDFS
    def ensure_hdfs_directories(spark, paths):
        try:
            fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
            for path in paths:
                hdfs_path = spark._jvm.org.apache.hadoop.fs.Path(path)
                if not fs.exists(hdfs_path):
                    fs.mkdirs(hdfs_path)
                    logger.info(f"Created HDFS directory: {path}")
                else:
                    logger.info(f"HDFS directory already exists: {path}")
        except Exception as e:
            logger.error(f"ERROR ensuring HDFS directories: {e}")
            raise

    ensure_hdfs_directories(spark, [HDFS_RAW_DATA_PATH, CHECKPOINT_LOCATION_HDFS])

    # Đọc dữ liệu từ Kafka
    try:
        kafka_df = spark \
            .readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", KAFKA_BROKER) \
            .option("subscribe", KAFKA_TOPIC) \
            .option("startingOffsets", "earliest") \
            .option("failOnDataLoss", "false") \
            .load()
        logger.info("Kafka stream loaded.")
    except Exception as e:
        logger.error(f"ERROR loading Kafka stream: {e}")
        spark.stop()
        exit()

    # Parse JSON từ Kafka
    schema = StructType([
        StructField("crawl_timestamp", LongType(), True),
        StructField("price_date", StringType(), True),
        StructField("gold_type", StringType(), True),
        StructField("buy_price", IntegerType(), True),
        StructField("sell_price", IntegerType(), True),
        StructField("source", StringType(), True)
    ])
    parsed_df = kafka_df.selectExpr("CAST(value AS STRING) as json_value") \
                        .select(from_json(col("json_value"), schema).alias("data")) \
                        .select("data.*")

    # Thêm cột để partition trên HDFS
    parsed_df = parsed_df \
        .withColumn("year", year(to_date(col("price_date"), "yyyy-MM-dd"))) \
        .withColumn("month", month(to_date(col("price_date"), "yyyy-MM-dd"))) \
        .withColumn("day", dayofmonth(to_date(col("price_date"), "yyyy-MM-dd")))

    # Ghi dữ liệu vào HDFS dưới dạng Parquet
    try:
        query = parsed_df \
            .writeStream \
            .format("parquet") \
            .outputMode("append") \
            .partitionBy("year", "month", "day") \
            .option("path", HDFS_RAW_DATA_PATH) \
            .option("checkpointLocation", CHECKPOINT_LOCATION_HDFS) \
            .trigger(processingTime='1 minute') \
            .start()

        logger.info(f"Writing stream to HDFS path '{HDFS_RAW_DATA_PATH}'...")
        query.awaitTermination()
    except Exception as e:
        logger.error(f"ERROR writing to HDFS: {e}")
        spark.stop()
        exit()
    finally:
        spark.stop()
        logger.info("Spark Streaming job finished.")

if __name__ == "__main__":
    main()