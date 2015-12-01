"""
Processes data
arguments: (infile, outfile, filt=true, filterval=250.0, average=true, split=false, splitval=20.0)
infile: string - name of file to process
outfile: string - name of file to output. If split=true, this name will have
    less and more prefixes for the two files
filt: boolean - whether to remove rows based on final value
filterval: float - what value to remove above
average: boolean - whether to average all hour measurements into one
split: boolean - whether to split into two files
splitval: float - what reflectivity value to split at if split = true
"""

import csv
import numpy

#big csvs require changing maximum size allowed: http://lethain.com/handling-very-large-csv-and-xml-files-in-python/
csv.field_size_limit(1000000000)

def processor(infile, outfile, filt=True,filterval=250.0, average=True,split=False,splitval=20.0):
    inputreader = csv.reader(open(infile,'r'), delimiter=",")
    if split:
        lesswriter = csv.writer(open("less"+outfile,'w'),delimiter=",")
        morewriter = csv.writer(open("more"+outfile,'w'),delimiter=",")
    else:
        outwriter = csv.writer(open(outfile,'w'),delimiter=",")
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
                    if ((not filt) or (float(idray[-1][-1]) <= filterval)):
                        #using this as cutoff point not incorrect data, since record for rainfall in US
                        #http://www.wunderground.com/blog/weatherhistorian/what-is-the-most-rain-to-ever-fall-in-one-minute-or-one-hour
                        #average or not?
                        if average:
                            newrow = []
                            idray = numpy.array(idray)
                            for i in range(len(idray[0])):
                                col = idray[:,i]
                                temp = [float(j) for j in col if (len(j) > 0)]
                                newrow.append(numpy.mean(temp))
                            if split:
                                if (newrow[3] >= splitval):
                                    morewriter.writerow(newrow)
                                else:
                                    lesswriter.writerow(newrow)
                            else:
                                outwriter.writerow(newrow)
                        else:
                            if split:
                                if (float(idray[-1][3]) >= splitval):
                                    morewriter.writerows(idray)
                                else:
                                    lesswriter.writerows(idray)
                            else:
                                outwriter.writerows(idray)
                except ValueError:
                    #wasn't a number, just ignore
                    pass
            #now reset for next id
            idray = []
            idray.append(row)
        prev = row
    #have to process the last ID afterward
    if (not filt or (float(idray[-1][-1]) <= filterval)):
        if average:
            newrow = []
            idray = numpy.array(idray)
            for i in range(len(idray[0])):
                col = idray[:,i]
                temp = [float(j) for j in col if (len(j) > 0)]
                newrow.append(numpy.mean(temp))
            if split:
                if (newrow[3] >= splitval):
                    morewriter.writerow(newrow)
                else:
                    lesswriter.writerow(newrow)
            else:
                outwriter.writerow(newrow)
        else:
            if split:
                if (float(idray[-1][3]) >= splitval):
                    morewriter.writerows(idray)
                else:
                    lesswriter.writerows(idray)
            else:
                outwriter.writerows(idray)
