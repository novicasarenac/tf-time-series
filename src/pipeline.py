import os
import datetime
import tensorflow as tf
import tensorflow_model_analysis as tfma

from typing import Text, Dict
from tfx.orchestration.pipeline import Pipeline
from tfx.orchestration.airflow.airflow_dag_runner import AirflowDagRunner
from tfx.orchestration.airflow.airflow_dag_runner import AirflowPipelineConfig
from tfx.orchestration import metadata
from tfx.utils.dsl_utils import tfrecord_input
from tfx.components import ImportExampleGen
from tfx.components import StatisticsGen
from tfx.components import SchemaGen
from tfx.components import ExampleValidator
from tfx.components import Transform
from tfx.components import Trainer
from tfx.components import Evaluator
from tfx.components import Pusher
from tfx.components.base import executor_spec
from tfx.components.trainer.executor import GenericExecutor
from tfx.proto import example_gen_pb2
from tfx.proto import trainer_pb2
from tfx.proto import pusher_pb2


PIPELINE_NAME: Text = 'timeseries_forecasting'
AIRFLOW_HOME: Text = os.environ.get('AIRFLOW_HOME',
                                    os.path.join(os.environ.get('HOME'), 'airflow'))
DATA_PATH: Text = os.path.join(AIRFLOW_HOME, 'data')
TFX_ROOT: Text = os.path.join(AIRFLOW_HOME, 'tfx')
PIPELINE_ROOT: Text = os.path.join(TFX_ROOT, 'pipelines', PIPELINE_NAME)
METADATA_PATH: Text = os.path.join(TFX_ROOT, 'metadata', PIPELINE_NAME, 'metadata.db')
SERVING_MODEL_DIR: Text = os.path.join(AIRFLOW_HOME, 'serving_model')
LABEL_KEY: Text = 'label'

# Modules
TRANSFORM_MODULE: Text = os.path.join(AIRFLOW_HOME, 'dags', 'transform.py')
TRAINER_MODULE: Text = os.path.join(AIRFLOW_HOME, 'dags', 'trainer.py')

AIRFLOW_CONFIG: Dict = {
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

    trainer = Trainer(module_file=TRAINER_MODULE,
                      examples=transform.outputs['transformed_examples'],
                      schema=schema_gen.outputs['schema'],
                      transform_graph=transform.outputs['transform_graph'],
                      custom_executor_spec=executor_spec.ExecutorClassSpec(GenericExecutor),
                      train_args=trainer_pb2.TrainArgs(num_steps=200),
                      eval_args=trainer_pb2.EvalArgs(num_steps=35))

    model_spec = tfma.ModelSpec(label_key=LABEL_KEY)
    slicing_spec = tfma.SlicingSpec()

    value_threshold = tfma.GenericValueThreshold(upper_bound={'value': 0.7})
    threshold = tfma.MetricThreshold(value_threshold=value_threshold)
    metric_config = tfma.MetricConfig(class_name='MeanAbsoluteError',
                                      threshold=threshold)
    metrics_spec = tfma.MetricsSpec(metrics=[metric_config])

    eval_config = tfma.EvalConfig(model_specs=[model_spec],
                                  slicing_specs=[slicing_spec],
                                  metrics_specs=[metrics_spec])
    evaluator = Evaluator(examples=example_gen.outputs['examples'],
                          model=trainer.outputs['model'],
                          eval_config=eval_config)

    filesystem = pusher_pb2.PushDestination.Filesystem(base_directory=SERVING_MODEL_DIR)
    push_destination = pusher_pb2.PushDestination(filesystem=filesystem)
    pusher = Pusher(model=trainer.outputs['model'],
                    model_blessing=evaluator.outputs['blessing'],
                    push_destination=push_destination)

    pipeline = Pipeline(pipeline_name=pipeline_name,
                        pipeline_root=pipeline_root,
                        metadata_connection_config=metadata.sqlite_metadata_connection_config(metadata_path),
                        components=[example_gen,
                                    statistics_gen,
                                    schema_gen,
                                    example_validator,
                                    transform,
                                    trainer,
                                    evaluator,
                                    pusher],
                        enable_cache=True,
                        beam_pipeline_args=['--direct_num_workers=0'])
    return pipeline


DAG = AirflowDagRunner(AirflowPipelineConfig(AIRFLOW_CONFIG)).run(
    create_pipeline(pipeline_name=PIPELINE_NAME,
                    pipeline_root=PIPELINE_ROOT,
                    metadata_path=METADATA_PATH)
)
