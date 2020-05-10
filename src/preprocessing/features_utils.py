import tensorflow as tf
from typing import Text, List


def bytes_feature(value: Text) -> tf.train.Feature:
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def float_feature(value: float) -> tf.train.Feature:
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value: int) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def float_list_feature(value: List) -> tf.train.Feature:
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))
