# -*- coding: utf-8 -*-
import os
import shutil
import random
import time
from pathlib import Path
import numpy as np

def getList(ifile):

    with open(ifile, 'r') as f:
        ret = []
        for line in f:
            ret.append(line.replace('\n', ''))
    
    return ret


def getFeatures(fname):

    with open(fname) as f:
        return f.readline().strip().split('\t')


def normalizeData(X, avg, std):
    """Z-score 정규화"""
    if isinstance(avg, str):
        avg_arr = np.loadtxt(avg)
    else:
        avg_arr = avg

    if isinstance(std, str):
        std_arr = np.loadtxt(std)
    else:
        std_arr = std

    return (X - avg_arr) / std_arr


def binarize_probabilities(proba, threshold):
    
    return (proba >= threshold).astype(float)


def endProcess(work):
    
    shutil.rmtree(work)

def getRandomString(length):

    str1 = 'abcdefghijklmnopqrstuvwxyz'
    str2 = str1.upper()
    numbers = '0123456789'
    box = str1 + str2 + numbers

    r = ''
    for i in range(length):
        s = random.sample(box, 1)[0]
        r += s

    return r

def tempGenerator():

    base = time.ctime()
    mon = base[4:7]
    dat = base[8:10].replace(' ', '0')
    tim = base[11:19].replace(':', '')

    return mon + dat + tim + getRandomString(3)

def MergeProcess(UnPredmols, PredAble_result):

    for m in UnPredmols:
        PredAble_result[m] = str("Invalid SMILES")

    return PredAble_result
