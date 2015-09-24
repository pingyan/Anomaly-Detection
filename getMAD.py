#/usr/bin/python
import gzip
import subprocess
import csv
import os
import sys
import shutil
import numpy
import pprint
import copy
import re

from math import log
from os import walk
from pyspark import SparkContext, SparkConf, SparkFiles

from pyspark.sql import Row
from pyspark.sql import SQLContext
from pyspark.sql import *

def summaryStats(x):

        # ----------------------------------
        # calculate and return the min, the max, the std, and the MAD
        # ----------------------------------

        xx = x[1]

        xx = [('0',z[1]) if z[0]=='' else (z) for z in xx]
        # given value and its frequency, make a list of lists of these values
        t = [[z[0]]*int(z[1]) for z in xx]
        # flatten list of lists
        expended = [int(item) for iter_ in t for item in iter_]
        myMean = numpy.mean(expended)
        myMedian = numpy.median(expended)
        myMax = numpy.amax(expended)
        myStd = numpy.std(expended)
        myMad = numpy.median(numpy.abs([i-myMedian for i in expended]))
        return (x[0], myMean, myMedian, myMax, myStd, myMad)

'''
Feature Entropy: 
“In information theory, entropy is the average amount of information contained in each message received.” Entropy measures the randomness and thus predictability of a profiled feature. It is one of the profile variability measurement. 

The input is a collection of a feature tuples with a probability: (feature_key, feature_value, frequency, total, prob)
The entropy is defined as:  
H(feature_key) = Σ prob(feature_value)*log(prob(feature_value))
The higher is H(feature), the more volatile is the profiled feature, i.e., the more likely we will see an anomalous feature value against the profile.
'''

def freqDist(x):
        # ---------------------
        # Given a list of item and item frequencies,
        # calcuate ENTROPY and distribution of items and return them
        # ---------------------

        total = sum([z[1] for z in x])

        dist =[(z[0],z[1],z[1]*1.0/total)  for z in x]
        entropy = sum([-x[2]*log(x[2],2) for x in dist])
        return (entropy, dist)

def freqFeature(ll, feature):
        # ---------------------
        # Get feature frequency by Grouping on a composite key
        # ---------------------

        KV =ll.map(lambda s:((s[0],s[1]),1))
        KVfreq = KV.reduceByKey(lambda x,y:x+y)
        flatKVfreq = KVfreq.map(lambda (x, y): ((x[0],x[1]),(x[2],y))).groupByKey().cache()
        freqFeatureRDD = flatKVfreq.mapValues(freqDist)
        return freqFeatureRDD

def main(sc,sqlContext):

        ll = sc.textFile(os.path.join(LOG_DIR,"*.gz")).repartition(sc.defaultParallelism * 4)
        logfields = ll.map(lambda s: s.split('`')).cache()
	freqFeatureRDD = freqFeature(logfields, feature)

        statsRDD = freqFeatureRDD.mapValues(summaryStats)
	statsRDD.saveAsTextFile(OUT)

if __name__ == "__main__":

        sc = SparkContext(appName="Profiling")
        sqlContext=SQLContext(sc)

        main(sc,sqlContext)
