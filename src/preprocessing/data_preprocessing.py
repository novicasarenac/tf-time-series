import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import logging
from typing import Text, Tuple, List
from features_utils import (bytes_feature,
                            float_feature,
                            int64_feature)

logging.basicConfig(level=logging.INFO)


class DataPreprocessor:

    def __init__(self,
                 data_path: Text,
                 samples_offset: Text,
                 history_size: int) -> None:
        self.data_path = data_path
        self.samples_offset = samples_offset
        self.history_size = history_size

    def __call__(self, plot_features: bool) -> Tuple[np.array, np.array]:
        data = self._read_data()
        logging.info(f'Number of the samples in original dataset: {data.shape[0]}')
        resampled_dataset = data.resample(self.samples_offset)
        resampled_dataset = resampled_dataset.sum()
        logging.info(f'Number of the samples in resampled dataset: {resampled_dataset.shape[0]}')
        standardized_dataset = self._standardize(resampled_dataset.values)
        samples, labels = self._transform_to_samples(standardized_dataset)
        logging.info(f'Samples shape: {samples.shape}')
        logging.info(f'Labels shape: {labels.shape}')
        if plot_features:
            resampled_dataset.plot(subplots=True)
            plt.show()
        return (samples, labels)

    def _read_data(self) -> pd.DataFrame:
        dataset = pd.read_csv(self.data_path,
                              header=0,
                              sep=';',
                              low_memory=False,
                              infer_datetime_format=True,
                              parse_dates={'datetime': [0, 1]},
                              index_col=['datetime'])
        dataset = self._clean_dataset(dataset)
        return dataset

    def _clean_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset.replace('?', np.nan, inplace=True)
        dataset = dataset.astype('float32')
        self._fill_missing(dataset.values)
        return dataset

    def _fill_missing(self, dataset: np.array) -> None:
        one_day = 60 * 24
        for row in range(dataset.shape[0]):
            for column in range(dataset.shape[1]):
                if np.isnan(dataset[row, column]):
                    dataset[row, column] = dataset[row - one_day, column]

    def _transform_to_samples(self, dataset: np.array) -> Tuple[np.array, np.array]:
        samples = []
        labels = []
        targets = dataset[:, 0]
        start_index = self.history_size
        end_index = len(dataset) - 1
        for i in range(start_index, end_index):
            indices = range(i - self.history_size, i)
            samples.append(dataset[indices])
            labels.append(targets[i])
        return (np.array(samples), np.array(labels))

    def _standardize(self, dataset: np.array) -> np.array:
        mean = dataset.mean(axis=0)
        std_dev = dataset.std(axis=0)
        standardized = (dataset - mean) / std_dev
        return standardized


class DataSerializer:

    def __init__(self, path: Text) -> None:
        self.path = path

    def serialize(self, samples: np.array, labels: np.array) -> None:
        with tf.io.TFRecordWriter(self.path) as writer:
            for i, sample in enumerate(samples):
                example = self._create_example(sample, labels[i])
                writer.write(example)

    def _create_example(self, sample: np.array, label: np.array) -> Text:
        serialized_sample = tf.io.serialize_tensor(sample)
        feature = {
            'sample': bytes_feature(serialized_sample),
            'label': float_feature(label)
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        return example.SerializeToString()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp',
                        '--data_path',
                        help='Path to the dataset.')
    parser.add_argument('-so',
                        '--samples_offset',
                        help='Samples offset for data resampling. D: daily, H: hourly.',
                        choices=['D', 'H'])
    parser.add_argument('-pf',
                        '--plot_features',
                        help='Plot all features.',
                        action='store_true')
    parser.add_argument('-hs',
                        '--history_size',
                        help='Number of samples to use for prediction.')
    parser.add_argument('-tp',
                        '--tfrecord_path',
                        help='Path to the output TFRecord file.')
    args = parser.parse_args()
    processor = DataPreprocessor(args.data_path,
                                 args.samples_offset,
                                 int(args.history_size))
    samples, labels = processor(args.plot_features)
    serializer = DataSerializer(args.tfrecord_path)
    serializer.serialize(samples, labels)
