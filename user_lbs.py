#用户位置分析
#!python3
# -*- coding: utf-8 -*-
import requests, time
from math import sin, asin, cos, sqrt, radians
import numpy
import matplotlib
import pylab
from scipy.cluster.vq import kmeans2, whiten

from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import pandas as pd

def haversine(lonlat1, lonlat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lat1, lon1 = lonlat1
    lat2, lon2 = lonlat2
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def clustering_by_dbscan_and_kmeans2(X):
    distance_matrix = squareform(pdist(X, (lambda u, v: haversine(u, v))))

    db = DBSCAN(eps=2, min_samples=5, metric='precomputed')  # using "precomputed" as recommended
    y_db = db.fit_predict(distance_matrix)

    X['cluster'] = y_db

    results = {}
    for i in X.values:
        #避免当簇只有一个元素时产生的迭代问题
        if i[2] not in results.keys():
            results[i[2]] = [[i[1], i[0]]]
        else:
            if results[i[2]]:
                results[i[2]].append([i[1], i[0]])
            else:
                results[i[2]] = [[i[1], i[0]]]
    print ("DBSCAN output: ", len(results), results.keys())
    print ("KMeans calc center as below: ")
    
    for k in results.keys():
        xy = numpy.array(results[k])
        if len(xy)>1:
            z = numpy.sin(xy[:, 1] - 0.2 * xy[:, 1])
            z = whiten(z)
            zip_geo = list(zip(xy[:, 0], xy[:, 1], z))

            res, idx = kmeans2(numpy.array(zip_geo), 1, iter=20, minit='points')
            print (res) 

if __name__ == "__main__":
    X=pd.read_csv('user_lbs.csv',usecols=['geo_lat', 'geo_lng'])
    clustering_by_dbscan_and_kmeans2(X)

