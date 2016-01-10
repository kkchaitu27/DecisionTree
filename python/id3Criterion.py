"""
This module contains code for calculating the information gain of a dataset as defined by the ID3 (Information Theoretic) heuristic.
"""

import math
import pandas as pd
import numpy as np

def entropy(df, target):
    """
    Input : df-Pandas Data Frame and target-Target Attribute for which entropy has to be calculated
    Returns the entropy of the given data for the given target attribute.
    """

    value_frequency = df[target].value_counts()

    data_entropy = sum(-value_frequency*1.0/len(df) * np.log2(1.0*value_frequency/len(df)))
    #print data_entropy
        
    return data_entropy

def information_gain(df, attr, target):
    """
    Input : df-Pandas Data Frame, attr- attribute on which the information gain (reduction in entropy) has to be calculated and target-target attribute
    Returns Infomation gain calculated by splitting the data on the chosen attribute (attr).
    """
    attr_value_frequency = df[attr].value_counts()
    sum_counts = sum(attr_value_frequency)
    subset_entropy = 0.0

    # Calculation of sum of entropy for each subset of records weighted by their probability of occuring in the training set.
    for value in attr_value_frequency.index:
        value_prob = attr_value_frequency.get(value)/ sum_counts
        data_subset = df[df[attr]==value]
        subset_entropy += value_prob * entropy(data_subset, target)


    # Calculation of information gain
    return (entropy(df, target) - subset_entropy)

