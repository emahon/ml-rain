"""
Processes training data
removes rainfall > 250 mm, averages all measurements done in same hour
""" 

import csv
import numpy

#big csvs require changing maximum size allowed: http://lethain.com/handling-very-large-csv-and-xml-files-in-python/
csv.field_size_limit(1000000000)

inputfile = open("train.csv",'r')
inputreader = csv.reader(inputfile, delimiter=",")
outwriter = csv.writer(open("average.csv",'w'),delimiter=",")
prev = [-1]
idray = []
for row in inputreader:
    #want to clump together ones with the same ID
    if (row[0] == prev[0]):
        idray.append(row)
    else:
        #process the previous id before running through this one
        if (len(idray) > 0):
            try:
                if (float(idray[-1][-1]) <= 250):
                    #using this as cutoff point not incorrect data, since record for rainfall in US
                    #http://www.wunderground.com/blog/weatherhistorian/what-is-the-most-rain-to-ever-fall-in-one-minute-or-one-hour
                    #compress data into one row by averaging
                    newrow = []
                    idray = numpy.array(idray)
                    for i in range(len(idray[0])):
                        col = idray[:,i]
                        temp = [float(j) for j in col if (len(j) > 0)]
                        newrow.append(numpy.mean(temp))
                    outwriter.writerow(newrow)
            except ValueError:
                #wasn't a number, just ignore
                pass
        #now reset for next id
        idray = []
        idray.append(row)
    prev = row
