import csv
def csv_write(path,data):
    f = open(path, 'a') 
    writer = csv.writer(f)
    writer.writerow(data)
    f.close()


def csv_writerows(path,data):
    f = open(path, 'w') 
    writer = csv.writer(f)
    writer.writerows(data)
    f.close()