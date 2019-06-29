#-*- coding:utf-8 -*-
'''
Created on 2019年6月29日

@author: nier
'''
import numpy as np
from distance import Distance_Ghost
class cluster(object):
    def __init__(self, data=None):
        self.data = data
    
    def add_data(self, new_data):
        self.data = np.vstack((self.data, new_data))

class cluster_manager(object):
    """簇管理类
    Attr:
        cluster_list：簇的列表，里面存放每一次合并之后的簇
        dis：距离类，里面定义了选择用来计算样本距离的函数参数，以及用来计算簇距离的函数的参数
        dis_matrix：距离矩阵，里面的顺序和cluster_list相同，（i，j）表示第i个簇和第j个簇的距离
    """
    def __init__(self):
        self.clusters_list = []
        self.dis = Distance_Ghost()
        self.dis_matrix = []
        
    def init_data(self, data):
        """初始化数据，将每个数据都初始化为一个簇
        Args:
            data：数据
        """
        for i in range(data.shape[0]):
            '''
            ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            #这里之所以不直接调用cluster(data[i])而是调用cluster(np.array([data[i]]))，目地是为了将单个数据也包装成[[1,2,3,4]]这样的形式，而不是[1,2,3,4]这样的形式，这样的形式能够统一后面的操作
            #因为后面会代用np.vstack函数，这个函数如果合并[1,2,3,4]和[5,6,7,8]会变成[[1,2,3,4],[5,6,7,8]]到最后，如果存在只有一个数据的cluster的时候那么各个cluster的数据存储形式会是：
            #[[1,2,3,4],[5,6,7,8]]以及[1,2,3,4]，这样的话两种数据的形式就不同了，后面的操作就变的更加麻烦，因此这里首先将数据统一为[[]]形式
            ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            '''
            self.clusters_list.append(cluster(np.array([data[i]])))
    
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
        self.dis.set_dis(samples_dis_type, samples_dis_args, samples_dis_customize, cluster_dis_type, cluster_dis_args, cluster_dis_customize_fun)
        
    def cal_dis_matrix(self):
        """自动计算距离矩阵
        """
        if len(self.clusters_list) <= 0:
            return
        
        for i in range(len(self.clusters_list)):
            dis_list = []
            for j in range(len(self.clusters_list)):
                if i==j:
                    dis_list.append(float("inf"))
                else:
                    dis = self.dis.get_clusters_distance(self.clusters_list[i].data, self.clusters_list[j].data)
                    dis_list.append(dis)
            self.dis_matrix.append(dis_list)
    
    def get_minuse_in_dis_matrix_index_i_j(self):
        """获取两个最接近的簇的下标
        Retuens:
            i,j:两个最接近的簇的下标，其中保证i > j
        """
        cur_index_i = 0
        cur_index_j = 0
        cur_dis = float("inf")
        i = 0
        j = 0
        while i < len(self.clusters_list):
            j = i + 1
            while j < len(self.clusters_list):
                if cur_dis > self.dis_matrix[i][j]:
                    cur_dis = self.dis_matrix[i][j]
                    cur_index_i = i
                    cur_index_j = j
                j = j + 1
            i = i + 1
        return cur_index_i, cur_index_j 
    
    def merge_2_cluster(self, i, j):
        """合并两个簇
        Args:
            i：第一个簇在cluster_list中的下标
            j：第二个簇在cluster_list中的下标
        """
        
        self.clusters_list[i].add_data(self.clusters_list[j].data)
        #将最后一个簇放到第j个簇的位置，然后在删除最后一个簇
        self.clusters_list[j] = cluster(self.clusters_list[len(self.clusters_list) - 1].data)
        
        #将距离矩阵中最后一个簇与其他簇的距离数据放到第j个簇的未知数据上，覆盖第j个簇的数据
        for index in range(len(self.clusters_list)):
            self.dis_matrix[index][j] = self.dis_matrix[index][len(self.clusters_list) - 1]
            self.dis_matrix[j][index] = self.dis_matrix[len(self.clusters_list) - 1][index]
        self.clusters_list.pop(len(self.clusters_list) - 1)
        
        #重新计算第i个簇和其他数据的距离，行和列都要改变
        for index in range(len(self.clusters_list)):
            self.dis_matrix[index][i] = self.dis.get_clusters_distance(self.clusters_list[index].data, self.clusters_list[i].data)
            self.dis_matrix[i][index] = self.dis.get_clusters_distance(self.clusters_list[i].data, self.clusters_list[index].data)
    
    def train(self,data_x, cluster_quantity = 1,samples_dis_type=None, samples_dis_args=None, samples_dis_customize=None,
                cluster_dis_type=None, cluster_dis_args=None, cluster_dis_customize_fun=None):
        """
        Args:
            data_x：数据项
            cluster_quantity：最终分类类的个数
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
        self.init_data(data_x)
        self.set_dis(samples_dis_type, samples_dis_args, samples_dis_customize, cluster_dis_type, cluster_dis_args, cluster_dis_customize_fun)
        self.cal_dis_matrix()
        while(len(self.clusters_list) > cluster_quantity):
            i, j = self.get_minuse_in_dis_matrix_index_i_j()
            self.merge_2_cluster(i, j)