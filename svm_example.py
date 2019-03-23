# coding:utf-8
# TODO: the first python script for svm classifier using opencv3

import cv2
import numpy as np
import os


def getImg_valid_max(img,th):
    '''
    TODO: using cumulative distribution for searching effective(valid) maximum
    :param img: input img readed by cv2
    :param th: the threshold of cdf
    :return: effective max
    '''
    hist,bins = np.histogram(img,bins=256)
    cdf = hist.cumsum() / (img.shape[0]* img.shape[1])
    index = np.where(cdf < th)
    ind = index[0]
    ind = ind[-1]
    valid_max = int(bins[ind])
    return valid_max

def getEdgeAverage(img,ratio):
    '''
    TODO: calculate the average of image edge region gray intensity
    :param img: input img
    :param ratio: the ratio of edge area to total area
    :return: four edge area`s mean of intensity
    '''
    row = img.shape[0]
    col = img.shape[1]

    dis = int(np.sqrt(col*row*ratio))    # edge area side length

    m1 = np.mean(img[0:dis,0:dis])
    m2 = np.mean(img[row-dis:row,0:dis])
    m3 = np.mean(img[0:dis,col-dis:col])
    m4 = np.mean(img[row-dis:row,col-dis:col])

    return (m1,m2,m3,m4)

def getCenterAverage(img,ratio):
    '''
    TODO: calculate center region average of intensity
    :param img: input img
    :param ratio:  ratio of center area to total area
    :return: mean of intensity
    '''
    row = img.shape[0]
    col = img.shape[1]

    ind_row = int(row/2)
    ind_col = int(col/2)

    dis = int(np.sqrt(col*row*ratio)/2)

    mean_center = np.mean(img[ind_row-dis:ind_row+dis,ind_col-dis:ind_col+dis])
    return mean_center

def normalize(data,max):
    data = np.array(data)
    return data/max


# fill positive samples

def train(path_pos,path_neg,path_save):
    '''
    TODO: using opencv SVM for pos-neg classify
    :param path_pos: positive samples path
    :param path_neg: negtive samples path
    :param path_save: saving model path
    :return:
    '''
    samples = []
    labels = []

    # get positive samples
    list_pos = os.listdir(path_pos)
    num = 0 # number of samples
    for file in list_pos:
        img_pos = cv2.imread(os.path.join(path_pos,file),cv2.IMREAD_UNCHANGED)

        ratio_edge = 0.02   # hyper parameter
        ratio_center = 0.04 # hyper parameter
        th = 0.99 # hyper parameter

        m1,m2,m3,m4 = getEdgeAverage(img_pos,ratio_edge)
        m_center = getCenterAverage(img_pos,ratio_center)
        valid_max = getImg_valid_max(img_pos,th)

        norm_center,norm_m1,norm_m2,norm_m3,norm_m4 = normalize((m_center,m1,m2,m3,m4),valid_max)
        samples.append(norm_center)
        samples.append(norm_m1)
        samples.append(norm_m2)
        samples.append(norm_m3)
        samples.append(norm_m4)
        num += 1
        labels.append(1.0)

    # get negtive samples
    list_neg = os.listdir(path_neg)
    for file in list_neg:
        img_neg = cv2.imread(os.path.join(path_neg, file), cv2.IMREAD_UNCHANGED)

        ratio_edge = 0.02  # hyper parameter
        ratio_center = 0.04  # hyper parameter
        th = 0.99  # hyper parameter

        m1, m2, m3, m4 = getEdgeAverage(img_neg, ratio_edge)
        m_center = getCenterAverage(img_neg, ratio_center)
        valid_max = getImg_valid_max(img_neg, th)

        norm_center, norm_m1, norm_m2, norm_m3, norm_m4 = normalize((m_center, m1, m2, m3, m4), valid_max)
        samples.append(norm_center)
        samples.append(norm_m1)
        samples.append(norm_m2)
        samples.append(norm_m3)
        samples.append(norm_m4)
        num += 1
        labels.append(0.0)

    samples = np.array(samples,dtype=np.float32).reshape(num,5) # get all the samples

    # shuffle sample
    rand = np.random.RandomState(234)   # generate random seed
    shuffle = rand.permutation(num)
    samples = samples[shuffle]
    labels = labels[shuffle]

    # create SVM classifier
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)   # set Type
    svm.setKernel(cv2.ml.SVM_RBF)   # set kernel
    svm.setGamma(5.383)
    svm.setC(2.67)

    # train
    svm.train(samples,cv2.ml.ROW_SAMPLE,labels) # every row is sample
    svm.save('svm_data.dat')






# if __name__ =="__main__":
    # list_pos = os.listdir('G:\ImageProcess\outlung\src')
    # samples = []
    # labels = []
    # num = 0
    # for file in list_pos:
    #     img_pos = cv2.imread(os.path.join('G:\ImageProcess\outlung\src', file), cv2.IMREAD_UNCHANGED)
    #
    #     ratio_edge = 0.02  # hyper parameter
    #     ratio_center = 0.04  # hyper parameter
    #     th = 0.99  # hyper parameter
    #
    #     m1, m2, m3, m4 = getEdgeAverage(img_pos, ratio_edge)
    #     m_center = getCenterAverage(img_pos, ratio_center)
    #     valid_max = getImg_valid_max(img_pos, th)
    #
    #     norm_center, norm_m1, norm_m2, norm_m3, norm_m4 = normalize((m_center,m1,m2,m3,m4), valid_max)
    #     samples.append(norm_center)
    #     samples.append(norm_m1)
    #     samples.append(norm_m2)
    #     samples.append(norm_m3)
    #     samples.append(norm_m4)
    #     num += 1
    #     labels.append(1.0)
    # samples = np.array(samples,dtype=np.float32).reshape(num, 5)
    # rand = np.random.RandomState(234)  # generate random seed
    # shuffle = rand.permutation(num)
    # print("samples ori")
    # print(samples)
    # samples = samples[shuffle]
    #
    # print(samples)

