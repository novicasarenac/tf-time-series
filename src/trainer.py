import tensorflow as tf
import tensorflow_transform as tft
from typing import Text, List, Tuple, Callable
from tfx.components.trainer.executor import TrainerFnArgs
from tensorflow_metadata.proto.v0.schema_pb2 import Schema


_BATCH_SIZE: int = 32
_LABEL: Text = 'label'
_FEATURES: List[Text] = ['sample']
_SAMPLE_SHAPE: Tuple[int, int] = (15, 7)
_EPOCHS: int = 20


def _gzip_reader_fn(filenames: List[Text]) -> tf.data.TFRecordDataset:
    return tf.data.TFRecordDataset(filenames,
                                   compression_type='GZIP')


def _input_fn(file_pattern: List[Text],
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int) -> tf.data.Dataset:
    transformed_feature_spec = tf_transform_output.transformed_feature_spec().copy()
    dataset = tf.data.experimental.make_batched_features_dataset(file_pattern=file_pattern,
                                                                 batch_size=batch_size,
                                                                 features=transformed_feature_spec,
                                                                 reader=_gzip_reader_fn,
                                                                 label_key=_LABEL)
    return dataset


def _build_model() -> tf.keras.Model:
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=_SAMPLE_SHAPE, name=_FEATURES[0]),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')
    model.summary()
    return model


def _get_serve_tf_examples_fn(model: tf.keras.Model,
                              tf_transform_output: tft.TFTransformOutput) -> Callable:
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples: tf.train.Example) -> tf.Tensor:
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(_LABEL)
        parsed_features = tf.io.parse_example(serialized_tf_examples,
                                              feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        transformed_features.pop(_LABEL, None)
        return model(transformed_features)

    return serve_tf_examples_fn


def run_fn(trainer_fn_args: TrainerFnArgs) -> None:
    tf_transform_output = tft.TFTransformOutput(trainer_fn_args.transform_output)
    train_dataset = _input_fn(trainer_fn_args.train_files,
                              tf_transform_output,
                              batch_size=_BATCH_SIZE)
    eval_dataset = _input_fn(trainer_fn_args.eval_files,
                             tf_transform_output,
                             batch_size=_BATCH_SIZE)
    model = _build_model()
    model.fit(train_dataset,
              epochs=_EPOCHS,
              steps_per_epoch=trainer_fn_args.train_steps,
              validation_data=eval_dataset,
              validation_steps=trainer_fn_args.eval_steps)

    tensor_spec = tf.TensorSpec(shape=[None],
                                dtype=tf.string,
                                name='examples')
    signatures = {
        'serving_default': (_get_serve_tf_examples_fn(model, tf_transform_output)
                            .get_concrete_function(tensor_spec))
    }
    model.save(trainer_fn_args.serving_model_dir,
               save_format='tf',
               signatures=signatures)
