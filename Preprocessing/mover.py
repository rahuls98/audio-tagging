import numpy as np
import pandas as pd
import os

train = pd.read_csv('fsd_test_targets.csv')
targets = train.columns[1:]
targets = list(targets)
#print(targets)

train_to_move = {}
for i in targets:
    label = i
    files = train.loc[train[label]==1]["fname"].to_list()
    train_to_move[label] = {}
    train_to_move[label]["files"] = files
    train_to_move[label]["count"] = len(files)

#print(train_to_move)

src_file = './FSDTest_images/'
dest_file = './dataset/'
command = 'cp '

""" for i in targets:
    label = i
    path = './dataset/' + label
    os.mkdir(path) """

for i in targets:
    #c = 0
    label = i
    dest = dest_file + label + '/'
    dest = '\"' + dest + '\"'
    files = train_to_move[label]['files']
    for j in files:
        filename = j.split('.')[0] + '.png'
        src = src_file + filename
        cmd = command + src + ' ' + dest
        #c += 1
        #print(cmd, c)
        try: os.system(cmd)
        except OSError: print("Creation for %s failed" %label)
        else: print("Successfully copied for %s " %label)

print("Done copying")