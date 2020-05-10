import os
import datetime

from typing import Text
from tfx.orchestration.pipeline import Pipeline
from tfx.orchestration.airflow.airflow_dag_runner import (AirflowDagRunner,
                                                          AirflowPipelineConfig)
from tfx.orchestration import metadata
from tfx.utils.dsl_utils import tfrecord_input
from tfx.components import ImportExampleGen
from tfx.components import StatisticsGen
from tfx.components import SchemaGen
from tfx.components import ExampleValidator
from tfx.components import Transform
from tfx.proto import example_gen_pb2


PIPELINE_NAME = 'timeseries_forecasting'
AIRFLOW_HOME = os.environ.get('AIRFLOW_HOME',
                              os.path.join(os.environ.get('HOME'), 'airflow'))
DATA_PATH = os.path.join(AIRFLOW_HOME, 'data')
TFX_ROOT = os.path.join(AIRFLOW_HOME, 'tfx')
PIPELINE_ROOT = os.path.join(TFX_ROOT, 'pipelines', PIPELINE_NAME)
METADATA_PATH = os.path.join(TFX_ROOT, 'metadata', PIPELINE_NAME, 'metadata.db')

# Modules
TRANSFORM_MODULE = os.path.join(AIRFLOW_HOME, 'dags', 'transform.py')


AIRFLOW_CONFIG = {
    'schedule_interval': None,
    'start_date': datetime.datetime(2020, 1, 1)
}


def create_pipeline(pipeline_name: Text,
                    pipeline_root: Text,
                    metadata_path: Text) -> Pipeline:
    # Read the dataset and split to train / eval
    output_config = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(splits=[
            example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=4),
            example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=1)
        ]))
    examples = tfrecord_input(DATA_PATH)
    example_gen = ImportExampleGen(input=examples,
                                   output_config=output_config)

    # Generate dataset statistics
    statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])

    # Generate schema based on statistics
    schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'],
                           infer_feature_shape=True)

    # Validate data and perform anomaly detection
    example_validator = ExampleValidator(statistics=statistics_gen.outputs['statistics'],
                                         schema=schema_gen.outputs['schema'])

    # Feature engineering
    transform = Transform(examples=example_gen.outputs['examples'],
                          schema=schema_gen.outputs['schema'],
                          module_file=TRANSFORM_MODULE)

    pipeline = Pipeline(pipeline_name=pipeline_name,
                        pipeline_root=pipeline_root,
                        metadata_connection_config=metadata.sqlite_metadata_connection_config(metadata_path),
                        components=[
                            example_gen,
                            statistics_gen,
                            schema_gen,
                            example_validator,
                            transform,
                        ],
                        enable_cache=True)
    return pipeline


DAG = AirflowDagRunner(AirflowPipelineConfig(AIRFLOW_CONFIG)).run(
    create_pipeline(pipeline_name=PIPELINE_NAME,
                    pipeline_root=PIPELINE_ROOT,
                    metadata_path=METADATA_PATH)
)
