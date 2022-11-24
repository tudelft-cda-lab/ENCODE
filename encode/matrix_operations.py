import numpy as np
import math
# import joblib
import pandas as pd
from tqdm import tqdm

def compute_row(feature_value, next_counts, previous_counts, row_length, next_offset, previous_offset, feature_frequency):
	'''
	Compute a row of in the matrix for the given feature value.
	It basically takes the frequency values of the bins and create
	corresponding vector as store it as a row in the matrix.
	'''
	row = np.zeros(row_length).tolist()
	row[0] = str(int(feature_value))
	row[1] = math.log(feature_frequency + 1)
	row[2] = next_counts[feature_value]['SELF']
	row[3] = next_counts[feature_value]['N']
	for i in range(0, len(next_counts[feature_value]) - 2):
		row[next_offset + i] = next_counts[feature_value][i]

	row[4] = previous_counts[feature_value]['SELF']
	row[5] = previous_counts[feature_value]['N']
	for i in range(0, len(previous_counts[feature_value]) - 2):
		row[previous_offset + i] = previous_counts[feature_value][i]

	return row

def compute_context_matrix(unique_feature_values, next_percentiles, previous_percentiles, next_counts, previous_counts, feature_frequencies):
	'''
	Compute the matrix that is used for the encoding.
	'''
	matrix = []
	column = ['symbol']
	column.append('log_frequency')
	column.append('self_next')
	column.append('next_nothing')
	column.append('self_previous')
	column.append('previous_nothing')
	next_offset = 6
	previous_offset = next_offset + len(next_percentiles)

	# for the next percentiles (header)
	for i in range(len(next_percentiles)):
		if i == 0:
			column.append('[0..' + str(next_percentiles[i]) + ']')
		else:
			column.append('[' + str(next_percentiles[i-1] + 1) + '..' + str(next_percentiles[i]) + ']')
	
	# for the previous percentiles (header)
	for i in range(len(previous_percentiles)):
		if i == 0:
			column.append('[0..' + str(previous_percentiles[i]) + ']')
		else:
			column.append('[' + str(previous_percentiles[i-1] + 1) + '..' + str(previous_percentiles[i]) + ']')

	matrix.append(column)

	# Currently using joblib does not provide any speedup than running everything on one single core or one single for loop.
	# # Generate the rows in parallel
	# rows = joblib.Parallel(n_jobs=-1, prefer='threads')(
	# 	joblib.delayed(compute_row) 
	# 	(unique_feature_values[k], next_counts, previous_counts, len(column), next_offset, previous_offset, feature_frequencies[unique_feature_values[k]]) 
	# 	for k in tqdm(range(len(unique_feature_values)))
	# )

	for j in tqdm(range(len(unique_feature_values))):
		matrix.append(
			compute_row(
				unique_feature_values[j], 
				next_counts,
			 	previous_counts, 
			 	len(column), 
			 	next_offset, 
			 	previous_offset, 
			 	feature_frequencies[unique_feature_values[j]])
		)
	
	return matrix


def load_matrix(path):
	'''
	Load a matrix that was computed previously. This is handy for 
	experimenting with different clustering methods or cluster numbers
	as the computation of the matrix is an expensive process.
	'''
	temp = pd.read_csv(path, delimiter=',').to_numpy()
	matrix = []
	row_values = []
	for i in range(len(temp)):
		row = temp[i]
		row_values.append(int(row[0]))
		matrix.append(row[1:])
	
	return matrix, row_values
