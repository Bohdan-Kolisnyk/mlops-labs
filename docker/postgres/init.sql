-- One server, three databases: Airflow metadata, MLflow backend store, our own warehouse
-- (raw, clean, features, prediction log). Runs once, when the data volume is first created.
CREATE DATABASE airflow;
CREATE DATABASE mlflow;
CREATE DATABASE warehouse;
