"""
K-means
Author : @Liyanzhe
2018.12.12
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Kmean():
    
    def __init__(self):
        print('A Kmeans Cluster~')

    def generate_examples(self, m:int, k:int, dim:int = 2):
        """
        generate 2 dimentional data examples, which is gaussian distribution.
        Input:
            m: Number of data points
            k: Number of clusters
        output:
            data: Numpy matrix  with shape(m,2).
        """
        datas = []
        centers = np.random.rand(k,dim) * 3 * k
        for i in range(k):
            this = np.random.randn(m, dim) + centers[i]
            datas.append(this)
        return np.concatenate(datas)

    def random_sample(self, data, k:int):
        """
        Randomly sample data point accroding given data.
        Input:
            data: numpy matrix with shape(m,n), where n is the dimention.
        return:
            samples: numpy matrix with shape(k,n), k is the number of samples
        """
        idx  = np.random.choice(data.shape[0], k, replace = False)
        samples = data[idx]
        return samples

    def plot(self, data, centers=None, labels=None, ax = None):
        """
        plot data According labels and center points
        Input:
            data: Numpy array with shape(m,2) where m is number of points
            centers: if None only plot data points
            labels : if None only plot center and points
        """
        x = data[:,0]
        y = data[:,1]
        if data.shape[1] == 2:
            if labels is not None:
                for c in range(centers.shape[0]):
                    plt.scatter(x[labels==c],y[labels==c])
            else:
                plt.scatter(x,y)
            if centers is not None:
                for c in centers:
                    plt.scatter(c[0], c[1], s=150, marker='x')
        elif data.shape[1] == 3:
            z = data[:,2]
            if labels is not None:
                for c in range(centers.shape[0]):
                    ax.scatter(x[labels==c], y[labels==c], z[labels==c])
            else:
                ax.scatter(x,y,z)
            if centers is not None:
                for c in centers:
                    ax.scatter(c[0],c[1],c[2], s=150, marker = 'x')


    def label_data(self, data, centers):
        """
        Label data with clusters accroding the distance betweend data adn centers 
        Input:
            data: Numpy matrix with shape(m,n)
            centers: Numpy centers with shape(k,n)
        Output:
            label: Numpy array with shape(m)
        """
        distance =  np.stack([np.linalg.norm(data-c, axis=1) for c in centers], axis=1)
        label = np.argmin(distance,axis=1)
        return label

    def solve(self, data, k: int, visualization = False):
        """
        the solution of data with kmeans clustering
        Input:
            data: numpy matrix with shape(m,n)
            k: the number of clusters
        Output:
            label: the label of clustered data
        """

        #################visual####
        if visualization:
            if data.shape[1] not in [2,3]:
                raise Exception('Invalid dimention', data.shape[1])
            fig = plt.figure()
            plt.ion()
            if data.shape[1] == 2:
                self.plot(data)
            else:
                ax = fig.add_subplot(111, projection="3d")
                ax.grid(False)
                ax._axis3don  = False
                self.plot(data, ax=ax)
            plt.waitforbuttonpress()
        #################visual####

        old_centers = None
        centers = self.random_sample(data, k)

        #################visual####
        if visualization:
            fig.clf()
            if data.shape[1] == 2:
                self.plot(data, centers)
            else:
                ax = fig.add_subplot(111, projection="3d")
                ax.grid(False)
                ax._axis3don  = False
                self.plot(data, centers, ax=ax)
            plt.waitforbuttonpress()

        while True:
            # compute the clusters of centers
            label = self.label_data(data, centers)

        #################visual####
            if visualization:
                fig.clf()
                if data.shape[1] == 2:
                    self.plot(data,centers,label)
                else:
                    ax = fig.add_subplot(111, projection="3d")
                    ax.grid(False)
                    ax._axis3don  = False
                    self.plot(data, centers,label, ax=ax)
                plt.waitforbuttonpress()
        #################visual####

            # save old centers for compare with new
            old_centers = centers.copy()
            # move centers
            for c in range(k):
                centers[c] = np.mean(data[label==c], axis=0)

        #################visual####
            if visualization:
                fig.clf()
                if data.shape[1] == 2:
                    self.plot(data, centers, label)
                else:
                    ax = fig.add_subplot(111, projection="3d")
                    ax.grid(False)
                    ax._axis3don  = False
                    self.plot(data, centers, label, ax=ax)
                plt.waitforbuttonpress()
        #################visual####

            # stop when centers don't move
            if (old_centers == centers).all() == True:
                if visualization:
                    plt.ioff()
                    plt.show()
                break

        return centers, label
if __name__ == "__main__":
    k = Kmean()
    data = k.generate_examples(100,3,3)
    centers, labels = k.solve(data, 3, visualization=True)