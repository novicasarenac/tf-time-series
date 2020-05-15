import tensorflow as tf
import tensorflow_transform as tft
import logging
from typing import Dict, List, Text, Tuple


logging.basicConfig(level=logging.INFO)


_FEATURES: List[Text] = ['sample']
_LABEL: Text = 'label'

_SAMPLE_SHAPE: Tuple[int, int] = (15, 7)


def parse_sample(sample: tf.Tensor) -> tf.Tensor:
    sample = tf.io.parse_tensor(sample, out_type=tf.float32)
    sample = tf.reshape(sample, _SAMPLE_SHAPE)
    return sample


def preprocessing_fn(inputs: Dict) -> Dict:
    outputs: Dict = {}
    for feature_name in _FEATURES:
        outputs[feature_name] = tf.compat.v2.map_fn(parse_sample,
                                                    tf.squeeze(inputs[feature_name], axis=1),
                                                    dtype=tf.float32)
    outputs[_LABEL] = inputs[_LABEL]
    return outputs
