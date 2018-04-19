'''
Transfer log to csv, then upload csv to plot.ly
'''
import sys
if __name__ == '__main__':
  if len(sys.argv) != 2:
    print("usage: python log2csv.py <logfilename>")
  log = open(sys.argv[1], 'r')
  csv = open('.'.join(sys.argv[1].split('.')[0:-1]) + ".csv", 'w')

  csv.write("epoch, avg_train_loss, train_eval_score, avg_dev_loss, dev_eval_score\n")
  for line in log:
    line = line.strip().split(" ")
    csv.write(','.join([line[6][:-1], line[8][:-1], line[10][:-1], line[12][:-1], line[14]]) + '\n')
  
  log.close()
  csv.close()
