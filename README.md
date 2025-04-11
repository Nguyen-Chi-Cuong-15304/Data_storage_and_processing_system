Mỗi lần down xong chạy lệnh : 
docker volume rm project_zk-data-h2dn project_zk-datalog-h2dn project_kafka-data-h2dn

#xoa checkpoint kafka_to_hdfs
docker exec namenode-h2dn hdfs dfs -rm -r /user/spark/checkpoints/kafka_to_hdfs_checkpoint


docker exec spark-master-h2dn /spark/bin/spark-submit --master spark://spark-master-h2dn:7077 --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.2.1 /opt/spark-apps/jobs/kafka_to_hdfs_raw.py


docker exec spark-master-h2dn /spark/bin/spark-submit --master spark://spark-master-h2dn:7077 --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.2.1,org.elasticsearch:elasticsearch-spark-30_2.12:8.9.0 /opt/spark-apps/jobs/kafka_to_es_stream.py


#tim noi chua noi dung
docker exec namenode-h2dn hdfs dfs -ls /user/data/gold_raw_parquet/year=2025/month=3/day=2 

#tra cuu noi dung
docker exec spark-master-h2dn /spark/bin/spark-submit --master spark://spark-master-h2dn:7077 /opt/spark-apps/jobs/view_parquet.py hdfs://namenode-h2dn:9000/user/data/gold_raw_parquet/year=2025/month=3/day=15/part-00000-ff54860f-0202-44b9-8809-ad807697c686.c000.snappy.parquet




