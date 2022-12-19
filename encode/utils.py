import math
import numpy as np
import pandas as pd
import csv

def find_percentile(value, threshold):
	"""
	For a given value, find the percentile it belongs to.
	"""
	for i in range(len(threshold)):
		if value <= threshold[i]:
			return i
	
	return -1

def compute_percentiles(feature_data, start, end, step):
	"""
	Compute the percentiles of a given feature.
	This functions allows the user to specify width of the percentiles.
	"""
	percentiles = []
	for i in range(start, end, step):
		percentiles.append(math.ceil(np.percentile(feature_data, i)))
	
	return percentiles

def get_sorting_level(level, columns_mapping):
	if level == 'src':
		return [columns_mapping['src_ip'], columns_mapping['timestamp']]
	elif level == 'dst':
		return [columns_mapping['dst_ip'], columns_mapping['timestamp']]
	elif level == 'conn':
		return [columns_mapping['src_ip'], columns_mapping['dst_ip'], columns_mapping['timestamp']]
	elif level == 'ts':
		return [columns_mapping['timestamp']]
	else:
		raise ValueError('Unknown level - make sure level is either src, dst, conn or ts')

def check_level(level, this_row, other_row, column_index_mapping):
	"""
	Check for two given rows if they are from the same level (src, dst, conn, ts).
	"""
	if level == 'src':
		return this_row[column_index_mapping['src_ip']] == other_row[column_index_mapping['src_ip']]
	elif level == 'dst':
		return this_row[column_index_mapping['dst_ip']] == other_row[column_index_mapping['dst_ip']]
	elif level == 'conn':
		return this_row[column_index_mapping['src_ip']] == other_row[column_index_mapping['src_ip']] and this_row[column_index_mapping['dst_ip']] == other_row[column_index_mapping['dst_ip']]
	elif level == 'ts':
		return True


def write_encoding_to_file(encoding, path):
	"""
	Write the compute encoding to a CSV file.
	"""
	print('Writing encoding to file...')
	with open(path, 'w') as output:
		output.write('symbol,encoding\n')
		for val in encoding:
			output.write(str(val) + ',' + str(encoding[val]) + '\n')


def read_csv_file(path, column_names):
	"""
	Reads a CSV file and returns a dataframe contain only data from the specified columns.
	We assume that the CSV follows the same format that was collected from the Elastic stack.
	It is possible to use other format as well, as long as the NetFlow data is stored as in
	the CSV format.
	"""
	data = pd.read_csv(path, delimiter=',')
	data = data.dropna()
	data = data[column_names]
	return data

def write_matrix_to_file(matrix, path):
	"""
	Write a given matrix to CSV file (with the header).
	"""
	print('Writing matrix to file...')
	with open(path, 'w') as f:
		csv_reader = csv.writer(f, delimiter=',')
		for row in matrix:
			csv_reader.writerow(row)

def find_column_index_mapping(columns_mapping, column_names):
	"""
	Find the mapping between column and its corresponding index
	"""
	column_index_mapping = dict()
	for f in columns_mapping:
		column_index_mapping[f] = column_names.index(columns_mapping[f])
	
	return column_index_mapping