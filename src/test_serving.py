import os
import requests
import json
import base64
import tensorflow as tf
from typing import Text


BASE_PATH: Text = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH: Text = os.path.join(BASE_PATH, 'data')
TFRECORD_PATH: Text = os.path.join(DATA_PATH, 'dataset.tfrecord')


if __name__ == '__main__':
    dataset = tf.data.TFRecordDataset([TFRECORD_PATH])
    for sample in dataset.take(1):
        encoded = base64.b64encode(sample.numpy())
        data = {
            'instances': [{'b64': encoded.decode('utf-8')}]
        }
        url = 'http://localhost:8501/v1/models/timeseries:predict'
        response = requests.post(url=url, data=json.dumps(data))
        print(response.content)
