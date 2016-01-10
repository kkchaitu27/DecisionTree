"""
This module contains functions to construct a decision tree for data classificiation.
"""

import pandas as pd
import numpy as np

def most_frequent(df,target):
    """
    Input : df-Pandas Data Frame and target-target attribute that holds classification value
    Returns the most frequent item in the data frame
    """
    most_frequent_value = df[target].value_counts().idxmax()
            
    return most_frequent_value

def get_attribute_values(df,attribute):
    """
    Input : df-Pandas Data Frame and attribute-attribute for which unique values have to be computed
    Returns the unique values of an attribute in the data frame
    """
    return df[attribute].unique()

def best_attribute(df, attributes, target, splitting_heuristic):
    """
    Iterates through all the attributes and returns the attribute with the best splitting_heuristic
    """
    best_value = 0.0 
    best_gain = 0.0
    best_attr = None

    for attr in attributes:
        gain = splitting_heuristic(df, attr, target)
        if (gain >= best_gain and attr != target):
            best_gain = gain
            best_attr = attr
                
    return best_attr

def create_decision_tree(df, attributes, target, splitting_heuristic_func):
    """
    Input : df-Pandas Data Frame, attributes-list of features, target-Target Feature, splitting_heuristic_func-Function to find best feature for splitting
    Returns a new decision tree based on the examples given.
    """
    vals = df[target]
    default = most_frequent(df, target)

    # If the dataset is empty or there are no independent features
    if target in attributes:
        attributes.remove(target)
    if df.empty or len(attributes) == 0:
        return default
    # If all the target values are same, return that classification value.
    elif len(vals.unique())==1:
        return default
    else:
        # Choose the next best attribute to best classify our data based on heuristic function
        best_attr = best_attribute(df, attributes, target,splitting_heuristic_func)

        # Create an empty new decision tree/node with the best attribute and an empty
        # dictionary object
        tree = {best_attr:{}}

        # Create a new decision tree/sub-node for each of the values in the best attribute field
        for val in get_attribute_values(df, best_attr):
            # Create a subtree for the current value under the "best" field
            data_subset = df[df[best_attr]==val]
            subtree = create_decision_tree(data_subset,
                [attr for attr in attributes if attr != best_attr],
                target,
                splitting_heuristic_func)

            # Add the new subtree to the empty dictionary object in our new
            # tree/node we just created.
            tree[best_attr][val] = subtree

    return tree

def get_prediction(data_row,decision_tree):
    """
    This function recursively traverses the decision tree and returns a  classification for the given record.
    """
    # If the current node is a string or integer, then we've reached a leaf node and we can return the classification
    if type(decision_tree) == type("string") or type(decision_tree) == type(1):
        return decision_tree

    # Traverse the tree further until a leaf node is found.
    else:
        attr = decision_tree.keys()[0]
        t = decision_tree[attr][data_row[attr]]
        return get_prediction(data_row, t)

def predict(tree,predData):
    """
    Input : tree-Tree Dictionary created by create_decision_tree function, predData-Pandas DataFrame on which predictions are made
    Returns a list of predicted values of predData
    """
    predictions = []
    for index, row in predData.iterrows():
       predictions.append(get_prediction(row, tree))


    return predictions
    


