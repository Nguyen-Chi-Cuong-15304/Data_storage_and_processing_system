
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col
import logging

# --- Cấu hình Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Cấu hình ---
HDFS_DATA_PATH = "hdfs://namenode-h2dn:9000/user/data/football_players_raw_parquet"
MODEL_PATH = "hdfs://namenode-h2dn:9000/user/spark/models/football_market_value_model"

def main():
    logger.info("Starting Spark ML job: Training model for Football Players")

    # Khởi tạo SparkSession
    spark = SparkSession \
        .builder \
        .appName("FootballPlayerMarketValuePrediction") \
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode-h2dn:9000") \
        .config("spark.hadoop.dfs.client.use.datanode.hostname", "true") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    logger.info("SparkSession created.")

    try:
        # Đọc dữ liệu từ HDFS
        logger.info(f"Reading data from HDFS path: {HDFS_DATA_PATH}")
        df = spark.read.parquet(HDFS_DATA_PATH)
        logger.info(f"Loaded {df.count()} records from HDFS.")

        # Chọn các cột đặc trưng và nhãn
        feature_columns = [
            "age", "appearances", "PPG", "goals", "assists", "own_goals",
            "substitutions_on", "substitutions_off", "yellow_cards",
            "second_yellow_cards", "red_cards", "penalty_goals",
            "minutes_per_goal", "minutes_played", "player_height"
        ]
        label_column = "market_value"

        # Kiểm tra dữ liệu null
        for col_name in feature_columns + [label_column]:
            null_count = df.filter(col(col_name).isNull()).count()
            if null_count > 0:
                logger.warning(f"Column {col_name} has {null_count} null values.")
                df = df.filter(col(col_name).isNotNull())

        # Lọc dữ liệu (loại bỏ thủ môn và strong_foot không hợp lệ)
        df = df.filter((col("position") != "Goalkeeper") & 
                       (col("strong_foot").isin("left", "right", "both")))
        logger.info(f"After filtering, {df.count()} records remain.")

        # Chuyển đổi dữ liệu thành vector đặc trưng
        assembler = VectorAssembler(
            inputCols=feature_columns,
            outputCol="features",
            handleInvalid="skip"  # Bỏ qua các hàng có giá trị không hợp lệ
        )
        feature_df = assembler.transform(df)
        logger.info("Features assembled.")

        # Chọn cột features và label
        final_df = feature_df.select("features", col(label_column).alias("label"))
        logger.info("Final DataFrame prepared for training.")

        # Chia dữ liệu thành tập huấn luyện và kiểm tra
        train_df, test_df = final_df.randomSplit([0.8, 0.2], seed=42)
        logger.info(f"Training set: {train_df.count()} records, Test set: {test_df.count()} records.")

        # Khởi tạo mô hình hồi quy tuyến tính
        lr = LinearRegression(featuresCol="features", labelCol="label")
        logger.info("LinearRegression model initialized.")

        # Huấn luyện mô hình
        lr_model = lr.fit(train_df)
        logger.info("Model training completed.")

        # Dự đoán trên tập kiểm tra
        predictions = lr_model.transform(test_df)
        logger.info("Predictions generated on test set.")

        # Đánh giá mô hình
        evaluator = RegressionEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="rmse"
        )
        rmse = evaluator.evaluate(predictions)
        logger.info(f"Root Mean Squared Error (RMSE) on test data: {rmse}")

        # Lưu mô hình vào HDFS
        logger.info(f"Saving model to {MODEL_PATH}")
        lr_model.write(MODEL_PATH)
        logger.info("Model saved successfully.")

        # Hiển thị một số dự đoán mẫu
        predictions.select("prediction", "label").show(5)

    except Exception as e:
        logger.error(f"ERROR during Spark ML job: {e}", exc_info=True)
    finally:
        logger.info("Stopping SparkSession...")
        spark.stop()
        logger.info("Spark ML job finished.")

if __name__ == "__main__":
    main()