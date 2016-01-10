import pandas as pd
import numpy as np
from decisiontree import *
from id3Criterion import *
import sys

def read_file():
    """
    Tries to read csv file using Pandas
    """
    # Get the name of the data file and load it into 
    if len(sys.argv) < 2:
        # Ask the user for the name of the file
        print "Filename: ", 
        filename = sys.stdin.readline().strip()
    else:
        filename = sys.argv[1]

    try:
        data = pd.read_csv(filename)
    except IOError:
        print "Error: The file '%s' was not found on this system." % filename
        sys.exit(0)

    return data

def print_tree(tree, str):
    """
    This function recursively crawls through the d-tree and prints it out in a
    more readable format than a straight print of the Python dict object.  
    """
    if type(tree) == dict:
        print "%s%s" % (str, tree.keys()[0])
        for item in tree.values()[0].keys():
            print "%s\t%s" % (str, item)
            print_tree(tree.values()[0][item], str + "\t")
    else:
        print "%s\t->\t%s" % (str, tree)

def build_tree(data):
    """
    This function builds tree from Pandas Data Frame with last column as the dependent feature
    """
    attributes = list(data.columns.values)
    target = attributes[-1]
    return create_decision_tree(data,attributes,target,information_gain)


if __name__ == "__main__":
    data = read_file()
    print "Data Read and Loaded\n"
    print "Building Decision Tree\n"
    tree = build_tree(data)
    print print_tree(tree,"")
    predictions = predict(tree,data)
    print predictions
