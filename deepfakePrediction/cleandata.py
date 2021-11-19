import pandas as pd
import sys
import os
import pprint

landmarks_x = ['x_{}'.format(x) for x in range(0,68)]
landmarks_y = ['y_{}'.format(y) for y in range(0,68)]

columns = ['filename', 'frame', 'confidence', 'face_id']+landmarks_x+landmarks_y+['label']

def makeChunks(i):
	chunksize = 200000

	j=0
	for chunk in pd.read_csv('dataset/train_dataset_{}.csv'.format(i), chunksize=chunksize):
		df = chunk[columns]
		df.to_csv('dataset/cleaned_data/file_{}_{}.csv'.format(i, j), header=columns, index=False)
		j+=1

def getCols(file):
	df = pd.read_csv(file, index_col=False)
	df = df[columns]
	df.to_csv(file, index=False)


if __name__ == "__main__":
	getCols(sys.argv[1])
	
