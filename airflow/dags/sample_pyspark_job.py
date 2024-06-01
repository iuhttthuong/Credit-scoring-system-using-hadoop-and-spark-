from airflow import DAG
from airflow.operators.bash_operator import BashOperator
import datetime as dt

# Định nghĩa lệnh shell để tải dữ liệu CSV lên HDFS
upload_command = """
docker exec -i namenode bash -c '
/opt/hadoop-3.2.1/bin/hdfs dfs -mkdir -p /home_credit/data
for file in /opt/data/*.csv; do
    filename=$(basename "$file")
    while [ ! -f "$file" ]; do
        echo "Waiting for $file to be fully copied..."
        sleep 5
    done
    # Kiểm tra nếu tệp vẫn đang được sao chép bằng cách kiểm tra phần mở rộng tạm thời
    while [ -f "$file._COPYING_" ]; do
        echo "File $file is still being copied..."
        sleep 5
    done
    /opt/hadoop-3.2.1/bin/hdfs dfs -put -f "$file" "hdfs://namenode:9000/home_credit/data/$filename"
done
'
"""


default_args = {
    'owner': 'FINAL',
    'start_date': dt.datetime.now() - dt.timedelta(minutes=19),
    'retries': 3,
    'retry_delay': dt.timedelta(minutes=1),
}

# Initialize DAG with your student ID as the name, run every 5 minutes
with DAG('final',
         default_args=default_args,
         tags=['final'],
         schedule_interval=dt.timedelta(minutes=10),
         ) as dag:
    upload_task = BashOperator(
        task_id='upload_csv_to_hdfs_task', bash_command=upload_command)

    run_spark = BashOperator(
        task_id='run_spark',
        bash_command="docker exec spark-master /spark/bin/spark-submit --master spark://spark-master:7077 /opt/scripts/pyspark/sample_spark_2.py")

    upload_task >> run_spark
