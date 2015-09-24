#/usr/bin/python
import glob
import os
import sys
import shutil

from math import log
from os import walk
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SQLContext
import numpy as np

'''
Median Absolute Deviation (MAD) Statistic

Outliers are computed as follows:
* Let X be all data points
* Let x_i be a data point in X
* Let MAD be the median absolute deviation, defined as
        MAD = median( for all x in X, | x - median(X)| )
* Let M_i be the modified z-score for payment x_i, defined as
        0.6745*(x_i - median(X))/MAD

As per the recommendations by the literature, a data point is
considered an outlier if the modified z-score, M_i > thresh, which
is 3.5 by default.

NOTE: regular MAD test evaluates on all points.
My modification compares the out-of-bag points with a pre-built univariate distribution
'''

def mad_based_outlier(point, med, thresh=3.5):
        #med_abs_deviation = profile.mad
        med = float(med)
        med_abs_deviation = med # med is the median of each training point's difference from their median
        diff = np.abs(point - med)
        if med_abs_deviation !=0:
                modified_z_score = 0.6745 * diff / med_abs_deviation
                "outlier" if modified_z_score > thresh else "normal"
                return modified_z_score > thresh
        else:
                return False

if __name__ == "__main__":
        sc = SparkContext("local", "Univariate anomoly test demo")
        ssc = StreamingContext(sc, 10)
        sqlc = SQLContext(sc)
        sqlc.setConf("spark.sql.shuffle.partition", "10")

        profile_X = sqlc.parquetFile(path_profile)
        mad = profile_X.first().mad

        test = ssc.textFileStream(dirpath_out_of_bag_datapoints)
        test = test.map(lambda x: x.split('`')[int(demo_numerical_field)])

        anomalousX = test.filter(lambda x: mad_based_outlier(int(x),mad))
        anomalousX.pprint()
