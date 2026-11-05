-- One server, four databases: Airflow metadata, MLflow backend store, our own warehouse
-- (raw, clean, features, prediction log) and the Feast registry + online store.
-- Runs once, when the data volume is first created (feature_repo/apply.py creates `feast` otherwise).
CREATE DATABASE airflow;
CREATE DATABASE mlflow;
CREATE DATABASE warehouse;
CREATE DATABASE feast;
