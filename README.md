# Anomaly-Detection

###Simple Univariate Statistical Deviation Tests for Anomaly Detection - MAD Test

Statistical deviation tests often involve calculating the absolute difference between a data value and it’s normalized mean or median with standard deviation and then a following a threshold test. In the Anomaly Detection context, deviation tests need to be tolerant of outliers, because the anomalous data points we try to detect would have influenced the typical estimation of mean and standard deviation. 

In fact, Median is more robust a measure of central tendency than mean and we can use the Median absolute deviation (MAD) as a robust measure of statistical dispersion.  

The Median Absolute Deviation is defined as below. 

MAD= median(|X-median(X)|)

This is call the modified Zccore, where a multiplier is often used, and the threshold for outlier detection is often set as 3.5.

P(0.6745 * (X-med(X))/MAD(X) > 3.5)

The MAD test allows us to detect outliers reliably even in the presence of outliers in the data used to compute median and median absolute deviation.

