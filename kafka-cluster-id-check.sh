#!/bin/bash

# Config
ZOOKEEPER_HOST="zookeeper-h2dn:2181"
KAFKA_DATA_DIR="/var/lib/kafka/data"

# Hàm lấy cluster ID từ Zookeeper
get_zookeeper_cluster_id() {
    # Sử dụng zookeeper-shell để lấy cluster ID
    echo "get /cluster/id" | zookeeper-shell "$ZOOKEEPER_HOST" | grep -Po '(?<=cluster.id=)[^\n]+' | tr -d '\r'
}

# Hàm lấy cluster ID từ Kafka meta.properties
get_kafka_cluster_id() {
    if [ -f "$KAFKA_DATA_DIR/meta.properties" ]; then
        grep -Po '(?<=cluster.id=).+' "$KAFKA_DATA_DIR/meta.properties"
    fi
}

# Chờ Zookeeper sẵn sàng
while ! nc -z zookeeper-h2dn 2181; do
    echo "Waiting for Zookeeper..."
    sleep 2
done

# Lấy cluster ID từ cả hai nguồn
ZK_ID=$(get_zookeeper_cluster_id)
KAFKA_ID=$(get_kafka_cluster_id)

echo "Zookeeper Cluster ID: $ZK_ID"
echo "Kafka Cluster ID:     $KAFKA_ID"

# Kiểm tra và xử lý mismatch
if [ -n "$ZK_ID" ] && [ -n "$KAFKA_ID" ] && [ "$ZK_ID" != "$KAFKA_ID" ]; then
    echo "Cluster ID mismatch detected! Resetting Kafka data..."
    rm -rf "$KAFKA_DATA_DIR"/*
fi