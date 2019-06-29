#-*- coding:utf-8 -*-
'''
Created on 2019年6月25日

@author: nier
'''
import numpy as np

class SampleDistance(object):
    """计算样本距离的类
    
    这个类不存储任何信息，也没有任何属性，仅仅实封装了计算两个样本之间距离的计算方法而已
    
    """
    def minkowski_distance(self, sample_1, sample_2, p = 2):
        """该函数用于定义一般的明可夫斯基距离，其中为了p表示范数，注意用p = -1的时候表示切比雪夫距离，既无穷大
        Args:
            sample_1,sample_2：需要计算距离的两个样本，现在限制为1维向量
        Return:
            返回sample_1,sample_2两个样本的明可夫斯基距离
        """
        new_sample = np.abs(sample_1 - sample_2)
        result = 0
        if p > 0:
            result = np.max(new_sample)
        else:
            new_sample = np.power(new_sample, p);
            result = np.power(new_sample.sum(), 1/p)
        
        return result
    
    def euclidean_distance(self, sample_1, sample_2):
        """该函数用于计算欧几里德距离
        Args:
        sample_1, sample_2:两个需要计算距离的样本
        """
        """#在使用下面的代码的时候，速度非常的慢，大概慢了2倍
        new_sample = np.abs(sample_1 - sample_2)
        new_sample = np.power(new_sample, 2)
        result =np.power(np.sum(new_sample), 1/2)
        """
        result = np.linalg.norm(sample_1-sample_2)
        
        return result
    
    def manhattan_distance(self, sample_1, sample_2):
        """计算曼哈顿距离
        Args:
            sample_1,sample_2:两个需要计算距离的样本
        """
        new_sample = np.abs(sample_1 - sample_2)
        return new_sample.sum()
    
    def chebyshev_distance(self, sample_1, sample_2):
        """计算切比雪夫距离
        Args:
            sample_1, sample_2:两个需要计算距离的样本
        """
        new_sample = sample_1 - sample_2
        return np.max(new_sample)
    
    def mahalanobis_distance(self, sample_1, sample_2, convariance_matrix):
        """目前求逆的算法暂时还不熟悉，所以暂时不完成
        """
        pass
    
    def correlation_coefficient(self, sample_1, sample_2):
        """相关系数计算，这里涉及到整个数据集的计算，因此需要仔细考虑其接口的合理性
        """
        pass
    
    def cosine(self, sample_1, sample_2):
        """余弦相似度计算
        Args:
            sample_1, sample_2:两个需要计算余弦相似度的样本
        """
        inner_product = np.dot(sample_1, sample_2)      #内积
        sample_1_mold = np.power(sample_1, 2).sum()     #sample_1的模长
        sample_2_mold = np.power(sample_2, 2).sum()     #sample_2的模长
        cosine_result = inner_product / np.power(sample_1_mold * sample_2_mold , 1/2)
        return cosine_result

    
class ClusterDistance(object):
    """计算两个类的距离，或者说是计算两个样本集合的距离
    注意这个类自身没有实现求两个样本之间的距离的函数，是接受一个对应的函数来进行计算的。即把计算接口交由外部实现，自身不实现该计算函数
    可以在初始化或者调用set函数设置
    """
    def __init__(self):
        self.get_samples_dis_fun = None
        self.samples_dis_type = None
        self.sample_dis_args = None
        self.sample_dis_customize_fun = None
    
    def set_samples_dis_fun(self, get_samples_dis_fun, samples_dis_type=None, samples_dis_args=None, samples_dis_customize_fun=None):
        self.get_samples_dis_fun = get_samples_dis_fun
        self.samples_dis_type = samples_dis_type
        self.sample_dis_args = samples_dis_args
        self.sample_dis_customize_fun = samples_dis_customize_fun
        
    def get_2_sample_distance(self, sample_1, sample_2):
        """调用外部赋值的计算样本距离的函数来进行计算。
        """
        dis = self.get_samples_dis_fun(sample_1,sample_2)
        return dis
        
    def single_linkage(self, sample_collection_1, sample_collection_2):
        """返回两个类的最短距离
        Args:
            sample_collection_1:类别1的数据集合
            sample_collection_2:类别2的数据集合
        Return:
            返回两个类之间的距离
        """
        cur_shortest_dis = 0
        for i in range(sample_collection_1.shape[0]):
            for j in range(sample_collection_2.shape[0]):
                dis = self.get_2_sample_distance(sample_collection_1[i], sample_collection_2[j])
                if cur_shortest_dis > dis:
                    cur_shortest_dis = dis
        
        return dis

    def complete_linkage(self, sample_collections_1, sample_collections_2):
        """返回两个类的最长距离
        Args:
            sample_collection_1:类别1的数据集合
            sample_collection_2:类别2的数据集合
        Retuen:
            返回两个类之间的最长距离
        """
        cur_longest_dis = 0
        for i in range(sample_collections_1.shape[0]):
            for j in range(sample_collections_2.shape[0]):
                dis = self.get_2_sample_distance(sample_collections_1[i], sample_collections_2[j])
                if cur_longest_dis < dis:
                    cur_longest_dis = dis
        
        return cur_longest_dis

    def center_distance(self, sample_collection_1, sample_collection_2):
        """返回两个类的中心的距离
        Args:
            sample_collection_1:类别1的数据集合
            sample_collection_2:类别2的数据集合
        Retuen:
            返回两个类之间中心的距离
        """
        center_1 = np.mean(sample_collection_1, axis=0)
        center_2 = np.mean(sample_collection_2, axis=0)
        dis = self.get_2_sample_distance(center_1, center_2)
        return dis

    def mean_distance(self, sample_collection_1, sample_collection_2):
        """返回两个类别的平均距离
        Args:
            sample_collection_1:类别1的数据集合
            sample_collection_2:类别2的数据集合
        Retuen:
            返回两个类之间中心的距离
        """
        average = 0
        total = 0
        for i in range(sample_collection_1.shape[0]):
            for j in range(sample_collection_2.shape[0]):
                total += 1
                average = average + (self.get_2_sample_distance(sample_collection_1[i], sample_collection_2[j]) - average) / total
        
        return average

    
class Distance_Ghost(object):
    
    def __init__(self, samples_dis_type=None, samples_dis_args=None, samples_dis_customize=None, 
                 cluster_dis_type=None, cluster_dis_args=None, cluster_dis_customize_fun=None):
        """构造函数，可以初始化计算两个样本距离的参数，可以初始化计算两个簇距离的参数
        Args:
            sample_dis_type：样本距离类型，字符串形式，如果没有赋值这个参数，则使用初始化或者set_dis函数时候的距离，否则默认是用欧几里德距离
                            Minkowski：明可夫斯基距离，distance_arg需要实一个正整数p
                            Manhattan：曼哈顿距离
                            Euclidean：欧几里德距离
                            Chebyshev：切比雪夫距离
                            Mahalanobis：马氏距离，distance_arg需要协方差矩阵
                            Correlation_coefficient：相关系数
                            Cosine：consin值
                            Customize：使用自定义函数，如果使用了这个参数，那么需要将自定义函数赋值给distance_customize_function，
                                        并且该函数的参数列表为（sample1,sample2,args），返回一个距离的浮点数或者整数，
                                        同时，还需要将该函数的参数赋值给distance_arg
            samples_dis_arg：计算距离时候需要的参数列表
            samples_dis_customize：当distance_type赋值为Customize的时候，就需要赋值该参数，该参数接受函数作为参数，且函数参数列表为（sample_1,sample_2,args），返回整数值或浮点数
            cluster_dis_type：簇距离类型，字符串形式，如果没有赋值这个参数，则使用初始化或者set_dis函数时候的距离，否则默认是使用最远距离
                            single_linkage：最近距离
                            complete_linkage：最远距离
                            center_distance：中心距离
                            mean_distance：平均距离
                            customize：使用自定义函数，如果使用了这个参数，那么需要将自定义函数赋值给cluster_dis_customize_fun，
                                        并且该函数的参数列表为（cluster_1,cluster_2,args），返回一个距离的浮点数或者整数，
                                        同时，还需要将该函数的参数赋值给cluster_dis_args
            cluster_dis_args：计算距离时候需要的参数列表
            dcluster_dis_customize：当cluster_dis_type赋值为customize的时候，就需要赋值该参数，该参数接受函数作为参数，且函数参数列表为（sample_1,sample_2,args），返回整数值或浮点数
        """
        if samples_dis_type == None:
            self.samples_dis_type = "euclidean"
        else:
            self.samples_dis_type = samples_dis_type
        self.samples_dis_args = samples_dis_args
        self.samples_dis_customize = samples_dis_customize
        if cluster_dis_type == None:
            self.cluster_dis_type = "complete_linkage"
        else:
            self.cluster_dis_type = cluster_dis_type
        self.cluster_dis_args = cluster_dis_args
        self.cluster_dis_customize_fun = cluster_dis_customize_fun
        self.cluster_dis = ClusterDistance()
        self.cluster_dis.set_samples_dis_fun(self.get_samples_distance, self.samples_dis_type, self.samples_dis_args, self.samples_dis_customize)
    
    def set_dis(self,samples_dis_type=None, samples_dis_args=None, samples_dis_customize=None,
                cluster_dis_type=None, cluster_dis_args=None, cluster_dis_customize_fun=None):
        """设置计算两个样本距离的参数，计算两个簇距离的参数
        Args:
            sample_dis_type：样本距离类型，字符串形式，如果没有赋值这个参数，则使用初始化或者set_dis函数时候的距离，否则默认是用欧几里德距离
                            Minkowski：明可夫斯基距离，distance_arg需要实一个正整数p
                            Manhattan：曼哈顿距离
                            Euclidean：欧几里德距离
                            Chebyshev：切比雪夫距离
                            Mahalanobis：马氏距离，distance_arg需要协方差矩阵
                            Correlation_coefficient：相关系数
                            Cosine：consin值
                            Customize：使用自定义函数，如果使用了这个参数，那么需要将自定义函数赋值给distance_customize_function，
                                        并且该函数的参数列表为（sample1,sample2,args），返回一个距离的浮点数或者整数，
                                        同时，还需要将该函数的参数赋值给distance_arg
            samples_dis_arg：计算距离时候需要的参数列表
            samples_dis_customize：当distance_type赋值为Customize的时候，就需要赋值该参数，该参数接受函数作为参数，且函数参数列表为（sample_1,sample_2,args），返回整数值或浮点数
            cluster_dis_type：簇距离类型，字符串形式，如果没有赋值这个参数，则使用初始化或者set_dis函数时候的距离，否则默认是使用最远距离
                            single_linkage：最近距离
                            complete_linkage：最远距离
                            center_distance：中心距离
                            mean_distance：平均距离
                            customize：使用自定义函数，如果使用了这个参数，那么需要将自定义函数赋值给cluster_dis_customize_fun，
                                        并且该函数的参数列表为（cluster_1,cluster_2,args），返回一个距离的浮点数或者整数，
                                        同时，还需要将该函数的参数赋值给cluster_dis_args
            cluster_dis_args：计算距离时候需要的参数列表
            dcluster_dis_customize：当cluster_dis_type赋值为customize的时候，就需要赋值该参数，该参数接受函数作为参数，且函数参数列表为（sample_1,sample_2,args），返回整数值或浮点数
        """
        if samples_dis_type == None:
            self.samples_dis_type = "euclidean"
        else:
            self.samples_dis_type = samples_dis_type
        self.samples_dis_args = samples_dis_args
        self.samples_dis_customize = samples_dis_customize
        if cluster_dis_type == None:
            self.cluster_dis_type = "single_linkage"
        else:
            self.cluster_dis_type = cluster_dis_type
        self.cluster_dis_args = cluster_dis_args
        self.cluster_dis_customize_fun = cluster_dis_customize_fun
        self.cluster_dis = ClusterDistance()
        self.cluster_dis.set_samples_dis_fun(self.get_samples_distance, self.samples_dis_type, self.samples_dis_args, self.samples_dis_customize)
        
    def get_samples_distance(self, sample_1, sample_2):
        """返回两个样本之间的距离，默认是欧几里德距离，可以使用set_dis函数设置想用的距离计算函数
        Args:
            sample1：样本1，一个n维向量
            sample2：样本2，一个n维向量
            """ 
        dis = 0
        if self.samples_dis_type == "minkowski":
            dis = SampleDistance().minkowski_distance(sample_1, sample_2, self.samples_dis_args)
        elif self.samples_dis_type == "manhattan":
            dis = SampleDistance().manhattan_distance(sample_1, sample_2)
        elif self.samples_dis_type == "chebyshev":
            dis = SampleDistance().chebyshev_distance(sample_1, sample_2)
        elif self.samples_dis_type == "mahalanobis":
            dis = SampleDistance().mahalanobis_distance(sample_1, sample_2, self.samples_dis_args)
        elif self.samples_dis_type == "correlation_coefficient":
            dis = SampleDistance().correlation_coefficient(sample_1, sample_2)
        elif self.samples_dis_type == "cosine":
            dis = SampleDistance().cosine(sample_1, sample_2)
        elif self.samples_dis_type == "customize":
            dis = self.samples_dis_customize(sample_1, sample_2, self.samples_dis_args)
        else:
            dis = SampleDistance().euclidean_distance(sample_1, sample_2)
        return dis
        
    def get_clusters_distance(self, cluster_1, cluster_2):
        """计算两个簇之间的距离，会使用set_dis或者初始化时候设置的参数选择的距离计算方式，如果都没设置，就使用默认参数：欧几里德距离，最短样本距离
        Args:
            cluster_1：簇1
            cluster_2：簇2
        Returns:
            两个簇之间的距离
        """
        
        dis = 0
        if self.cluster_dis_type == "single_linkage":
            dis = self.cluster_dis.single_linkage(cluster_1, cluster_2)
        elif self.cluster_dis_type == "complete_linkage":
            dis = self.cluster_dis.complete_linkage(cluster_1, cluster_2)
        elif self.cluster_dis_type == "center_distance":
            dis = self.cluster_dis.center_distance(cluster_1, cluster_2)
        elif self.cluster_dis_type == "customize":
            dis = self.cluster_dis_customize_fun(cluster_1,cluster_2, self.cluster_dis_args)
        else:
            dis = self.cluster_dis.mean_distance(cluster_1, cluster_2)
        return dis
    
    def isCluster(self, samples, cluster, samples_dis_type=None, samples_dis_args=None, samples_dis_customize_fun=None):
        """还没有实现.....
        """
        pass