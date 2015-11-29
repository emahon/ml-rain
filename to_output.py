"""
Takes a 1D numpy array and the file name you want
outputs a csv in the file format Kaggle wants
Example here: https://www.kaggle.com/c/how-much-did-it-rain-ii/data
Copy this into your directory so you can directly import it
"""

import csv

def to_output(array, name):
    outwriter = csv.writer(open(name,'w'),delimiter=",")
    i = 1
    #add header
    outwriter.writerow(["Id","Expected"])
    for row in array:
        outwriter.writerow([i,row])
        i += 1
