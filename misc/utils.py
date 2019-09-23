from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import time
import logging
import tensorflow as tf
import numpy as np
import pandas as pd
import argparse


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args


def makedirs(filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))


def get_logger(name, logpath, filepath, package_files=[],
               displaying=True, saving=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    log_path = logpath + name + time.strftime("-%Y%m%d-%H%M%S")
    makedirs(log_path)
    if saving:
        info_file_handler = logging.FileHandler(log_path)
        info_file_handler.setLevel(logging.INFO)
        logger.addHandler(info_file_handler)
    logger.info(filepath)
    with open(filepath, 'r') as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, 'r') as package_f:
            logger.info(package_f.read())
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)

    return logger


def find_trainable_variables(key):
    with tf.variable_scope(key):
        return tf.trainable_variables()


class Summarizer:
    def __init__(self, sess, config):
        self.sess = sess
        self.config = config
        self.summary_placeholders = {}
        self.summary_ops = {}
        self.summary_writer = tf.summary.FileWriter(self.config.summary_dir, self.sess.graph)

    # it can summarize scalars and images.
    def summarize(self, step, scope="", summaries_dict=None):
        """
        :param step: the step of the summary
        :param scope: variable scope
        :param summaries_dict: the dict of the summaries values (tag,value)
        :return:
        """
        summary_writer = self.summary_writer
        with tf.variable_scope(scope):

            if summaries_dict is not None:
                summary_list = []
                for tag, value in summaries_dict.items():
                    if tag not in self.summary_ops:
                        if len(value.shape) <= 1:
                            self.summary_placeholders[tag] = tf.placeholder('float32', value.shape, name=tag)
                        else:
                            self.summary_placeholders[tag] = tf.placeholder('float32', [None] + list(value.shape[1:]), name=tag)
                        if len(value.shape) <= 1:
                            self.summary_ops[tag] = tf.summary.scalar(tag, self.summary_placeholders[tag])
                        else:
                            self.summary_ops[tag] = tf.summary.image(tag, self.summary_placeholders[tag])

                    summary_list.append(self.sess.run(self.summary_ops[tag], {self.summary_placeholders[tag]: value}))

                for summary in summary_list:
                    summary_writer.add_summary(summary, step)
                summary_writer.flush()


class StatsLogger:
    def __init__(self):
        self.prediction_stats_2d_df = pd.DataFrame(columns=['Subject', 'Slice',
                                                'MeanContourPtsClass1', 'MeanContourPtsClass2', 'MeanContourPtsClass3',
                                                'MeanPtDistClass1', 'MeanPtDistClass2', 'MeanPtDistClass3',
                                                'AssdMeanToSamplesPerPointClass1', 'AssdMeanToSamplesPerPointClass2', 'AssdMeanToSamplesPerPointClass3',
                                                'DistMeanToGTPerPointClass1', 'DistMeanToGTPerPointClass2', 'DistMeanToGTPerPointClass3',
                                                'Dice2DClass1', 'Dice2DClass2', 'Dice2DClass3',
                                                'Assd2DClass1', 'Assd2DClass2', 'Assd2DClass3',
                                                'PairwiseDice2DClass1', 'PairwiseDice2DClass2', 'PairwiseDice2DClass3',
                                                'PairwiseAssd2DClass1', 'PairwiseAssd2DClass2', 'PairwiseAssd2DClass3',
                                                'DiceMeanToSamplesPerSliceClass1', 'DiceMeanToSamplesPerSliceClass2', 'DiceMeanToSamplesPerSliceClass3',
                                                'AssdMeanToSamplesPerSliceClass1', 'AssdMeanToSamplesPerSliceClass2', 'AssdMeanToSamplesPerSliceClass3',
                                                'PredAreaClass1', 'PredAreaClass2', 'PredAreaClass3',
                                                'GTAreaClass1', 'GTAreaClass2', 'GTAreaClass3'])

    def append(self, info, dice_mean_to_samples_per_slice, assd_mean_to_samples_per_slice, dice_per_slice, assd_per_slice):
        num_slices = dice_mean_to_samples_per_slice.shape[0]
        for s in range(num_slices):
            self.prediction_stats_2d_df = self.prediction_stats_2d_df.append({'Subject': info['Name'],
                                                                    'Slice': s,
                                                                    'Dice2DClass1': dice_per_slice[s, 1],
                                                                    'Dice2DClass2': dice_per_slice[s, 2],
                                                                    'Dice2DClass3': dice_per_slice[s, 3],
                                                                    'Assd2DClass1': assd_per_slice[s, 1],
                                                                    'Assd2DClass2': assd_per_slice[s, 2],
                                                                    'Assd2DClass3': assd_per_slice[s, 3],
                                                                    'DiceMeanToSamplesPerSliceClass1': dice_mean_to_samples_per_slice[s, 1],
                                                                    'DiceMeanToSamplesPerSliceClass2': dice_mean_to_samples_per_slice[s, 2],
                                                                    'DiceMeanToSamplesPerSliceClass3': dice_mean_to_samples_per_slice[s, 3],
                                                                    'AssdMeanToSamplesPerSliceClass1': assd_mean_to_samples_per_slice[s, 1],
                                                                    'AssdMeanToSamplesPerSliceClass2': assd_mean_to_samples_per_slice[s, 2],
                                                                    'AssdMeanToSamplesPerSliceClass3': assd_mean_to_samples_per_slice[s, 3]}, ignore_index=True)


    def save(self, filename):
        self.prediction_stats_2d_df.to_csv('{}.csv'.format(filename), index=False, na_rep='nan')
        self.prediction_stats_2d_df.to_pickle('{}.pkl'.format(filename))


def var_shape(x):
    out = [k.value for k in x.get_shape()]
    assert all(isinstance(a, int) for a in out), \
        "shape function assumes that shape is fully known"
    return out


def numel(x):
    return np.prod(var_shape(x))


def softmax(x, axis):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)