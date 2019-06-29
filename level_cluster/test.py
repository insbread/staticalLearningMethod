#-*- coding:utf-8 -*-
'''
Created on 2019年6月29日

@author: nier
'''
import numpy as np
from cluster import cluster_manager
import matplotlib.pyplot as plt
import time


start_time = time.clock()
data_x = np.random.randint(0,20,(100, 2))
cs = cluster_manager()
cs.train(data_x, 3,  cluster_dis_type="mean_distance")

print(len(cs.clusters_list))
print(cs.clusters_list[0].data.shape[0])
print(cs.clusters_list[0].data)
print(cs.clusters_list[1].data.shape[0])
print(cs.clusters_list[1].data)
print(cs.clusters_list[2].data.shape[0])
print(cs.clusters_list[2].data)

plt.scatter(cs.clusters_list[0].data[:,0], cs.clusters_list[0].data[:,1],color="red")
plt.scatter(cs.clusters_list[1].data[:,0], cs.clusters_list[1].data[:,1],color="blue")
plt.scatter(cs.clusters_list[2].data[:,0], cs.clusters_list[2].data[:,1],color="yellow")
plt.show()

print("times used:"+str(time.clock() - start_time))