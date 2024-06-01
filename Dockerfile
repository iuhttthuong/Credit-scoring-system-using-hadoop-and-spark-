FROM apache/airflow:2.7.3
RUN pip install apache-airflow==${AIRFLOW_VERSION}
# # Cài đặt Java
# USER root
# RUN apt-get update && apt-get install -y openjdk-11-jdk && apt-get clean

# Cài đặt các gói từ requirements.txt
COPY requirements.txt requirements.txt
# # Thiết lập quyền người dùng trở lại cho airflow
# USER airflow
RUN pip install -r requirements.txt

# # Thiết lập các biến môi trường cho Java và Spark
# ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk-amd64



