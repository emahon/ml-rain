import csv

#big csvs require changing maximum size allowed: http://lethain.com/handling-very-large-csv-and-xml-files-in-python/
csv.field_size_limit(1000000000)

inputfile = open("train.csv",'r')
inputreader = csv.reader(inputfile, delimiter=",")
lesswriter = csv.writer(open("less.csv",'w'),delimiter=",")
morewriter = csv.writer(open("more.csv",'w'),delimiter=",")
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
                    #now throw into one or another output file based on reflectivity
                    if (float(idray[-1][3]) >= 20):
                        morewriter.writerows(idray)
                    else:
                        lesswriter.writerows(idray)
            except ValueError:
                #wasn't a number, just ignore
                pass
        #now reset for next id
        idray = []
        idray.append(row)
    prev = row
