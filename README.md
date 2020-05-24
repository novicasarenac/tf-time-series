# tf-time-series
Time series forecasting pipeline using TensorFlow Extended (TFX). Components of the pipeline are orchestrated using Apache Airflow. All components are visualized in the interactive notebook.

## Setup and running

Requirements: `python 3.7` and `docker` (for model serving).

To install all dependencies, download and preprocess dataset and prepare Airflow DAG run:
```bash
./setup.sh
```

Run Airflow web server with:
```bash
airflow webserver -p <port>
```
Parameter `port` is a port where you want to run Airflow.

Open another terminal window and run a scheduler with:
```bash
airflow scheduler
```
Open a browser at: `http://localhost:<port>` and trigger a DAG, or trigger it with:
```bash
airflow trigger_dag timeseries_forecasting
```

## Model serving

When the pipeline is finished, the model is ready for serving. Set the `AIRFLOW_HOME` environment variable (the default home path is `~/airflow`):
```bash
export AIRFLOW_HOME=/path/to/airflow
```

Pull the Docker image for serving:
```bash
docker pull tensorflow/serving
```

Run Docker container:
```bash
docker run -p 8501:8501 --mount type=bind,source=${AIRFLOW_HOME}/serving_model,target=/models/timeseries -e MODEL_NAME=timeseries -t tensorflow/serving
```

Test model serving with:
```bash
python src/test_serving.py
```
