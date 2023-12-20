from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from collections import Counter
from encode.matrix_operations import compute_context_matrix, load_matrix
from encode.utils import *
from encode.frequency_utils import *

def process_data(data, level, feature, time_feature, time_host_column_mapping):
	"""
	Precompute all the necessary data for creating the encoding.
	"""
	results = dict()
	print('Sorting NetFlows based on given level')
	data = data.sort_values(by=get_sorting_level(level, time_host_column_mapping))

	print('Computing the unique values of the given feature...')
	results['unique_' + feature] = sorted(data[feature].unique())

	if not time_feature:
		results['unique_' + feature + '_frequency'] = Counter(data[feature])

	if time_feature:
		print('Feature is a time feature, computing percentiles...')
		results[feature + '_percentiles'] = compute_percentiles(data[feature], 5, 105, 5)

		print('Computing frequency of the percentiles...')
		raw_duration_to_percentiles = []
		for d in data[feature]:
			raw_duration_to_percentiles.append(find_percentile(d, results[feature + '_percentiles']))

		results['percentile_frequency'] = Counter(raw_duration_to_percentiles)

	# Computing the mapping a column name to its index in the DataFrame.
	# The indices are used for the computation of the context matrix.
	columns = data.columns.tolist()
	column_index_mapping = find_column_index_mapping(time_host_column_mapping, columns)
	column_index_mapping[feature] = columns.index(feature)
	
	print('Computing the counts of the next and previous symbols...')
	data_np = data.to_numpy()
	if not time_feature:
		next_counts, next_symbols = compute_next_counts(data_np, feature, level, column_index_mapping)
		results['next_' + feature + '_percentiles'] = compute_percentiles(next_symbols, 10, 110, 10)
		results['next_' + feature + '_counts'] = convert_symbols_to_percentiles(next_counts, results['next_' + feature + '_percentiles'])
		previous_counts, previous_symbols = compute_previous_counts(data_np, feature, level, column_index_mapping)
		results['previous_' + feature + '_percentiles'] = compute_percentiles(previous_symbols, 10, 110, 10)
		results['previous_' + feature + '_counts'] = convert_symbols_to_percentiles(previous_counts, results['previous_' + feature + '_percentiles'])
	elif time_feature: 
		next_counts, next_symbols = compute_next_counts_timed(data_np, feature, level, 0.5, column_index_mapping)
		results['next_' + feature + '_percentiles'] = compute_percentiles(next_symbols, 10, 110, 10)
		results['next_' + feature  + '_counts'] = convert_symbols_to_percentiles_timed(next_counts, results[feature + '_percentiles'], results['next_' + feature + '_percentiles'], True)
		previous_counts, previous_symbols = compute_previous_counts_timed(data_np, feature, level, column_index_mapping)
		results['previous_' + feature + '_percentiles'] = compute_percentiles(previous_symbols, 10, 110, 10)
		results['previous_' + feature + '_counts'] = convert_symbols_to_percentiles_timed(previous_counts, results[feature + '_percentiles'], results['previous_' + feature + '_percentiles'], False)

	return results

def compute_matrix(unique_values, next_percentiles, previous_percentiles, next_counts, previous_counts, feature_frequencyies, path):
	"""
	Construct the co-occurrence matrix that we use to store the context of each unique feature value.
	"""
	matrix = compute_context_matrix(
		unique_values, 
		next_percentiles,
		previous_percentiles,
		next_counts, 
		previous_counts,
		feature_frequencyies
		)

	write_matrix_to_file(matrix, path) # write matrix to file, might be handy for later
	matrix.pop(0) # remove header
	return [row[1:] for row in matrix] # we only use frequency values and not the actual symbols

def cluster_matrix_rows(matrix, num_clusters, kmean_runs):
	"""
	Cluster the rows of the context matrix that we have created for a given feature. The cluster 
	labels are used as the encoding for the feature values. We run K-Means multiple times as the 
	initialisation of K-means is random. The best run of K-Means is the one with the highest
	silhouette score.
	"""
	print('Preparing to cluster the rows of the matrix...')
	best_cluster_results = []
	best_cluster_score = -1337.0 # set the silhouette score to a large negative value (silhouette score is always in [-1, 1])
	for i in tqdm(range(kmean_runs), desc='Finding best clusters for encoding'):
		clusterer = KMeans(n_clusters=num_clusters)
		clusterer.fit(matrix)
		cluster_labels = clusterer.labels_
		score = silhouette_score(matrix, cluster_labels, metric='euclidean')
		if score > best_cluster_score:
			best_cluster_score = score
			print('New best score: ' + str(best_cluster_score))
			best_cluster_results = cluster_labels
		else:
			continue
	
	print('Done with clustering')
	return best_cluster_results

def create_encoding_mapping_for_int_values(unique_feature_values, cluster_labels):
	"""
	Create a mapping from the (integer) unique feature values to the corresponding cluster labels. This is the encoding
	that we use for the feature values.
	"""
	encoding_mapping = {}
	for i in range(len(unique_feature_values)):
		encoding_mapping[unique_feature_values[i]] = cluster_labels[i]
	
	return encoding_mapping


def create_encoding_mapping_for_float_values(float_values, percentiles, cluster_labels):
	"""
	Create a mapping from the percentiles to the corresponding cluster labels. This is specifically used for float
	values as we use percentiles for float values. This is the encoding that we use for the feature values.
	"""
	encoding_mapping = dict()
	percentile_to_cluster_mapping = dict()
	for i in range(len(percentiles)):
			percentile_to_cluster_mapping[i] = cluster_labels[i]
	
	for i in range(len(float_values)):
		encoding_mapping[float_values[i]] = percentile_to_cluster_mapping[find_percentile(float_values[i], percentiles)]

	return encoding_mapping

def encode(feature, time_host_column_mapping, time_feature=False, level='conn', kmean_runs=10, num_clusters=35, output_folder='./', data_path=None, data=None, precomputed_matrix=False, context_matrix_path=None):
	"""
	Encode a given feature of the NetFlow data.
	"""
	print('Reading and processing data...')
	if data_path is not None and data is None:
		data =  read_csv_file(data_path, list(time_host_column_mapping.values()) + [feature])
	else:
		data = data
	
	data_info = process_data(data, level, feature, time_feature, time_host_column_mapping)

	if precomputed_matrix:
		if context_matrix_path is None:
			raise ValueError('Must provide paths to the precomputed matrices if precomputed_matrix is set to True')

		print('Using precomputed matrix...')
		context_matrix, _ = load_matrix(context_matrix_path)
		
	else:
		print('Computing the matrices...')
		if not time_feature:
			context_matrix = compute_matrix(
				data_info['unique_' + feature],
				data_info['next_' + feature + '_percentiles'],
				data_info['previous_' + feature + '_percentiles'],
				data_info['next_' + feature + '_counts'],
				data_info['previous_' + feature + '_counts'],
				data_info['unique_' + feature + '_frequency'],
				output_folder + feature + '_context_matrix_' + level + '.csv',
			)
		else:
			context_matrix = compute_matrix(
				[x for x in range(20)],
				data_info['next_' + feature + '_percentiles'],
				data_info['previous_' + feature + '_percentiles'],
				data_info['next_' + feature + '_counts'],
				data_info['previous_' + feature + '_counts'],
				data_info['percentile_frequency'],
				output_folder + feature + '_context_matrix_' + level + '.csv',
			)

	if not time_feature:
		cluster_results = cluster_matrix_rows(context_matrix, num_clusters, kmean_runs)
		feature_encoding = create_encoding_mapping_for_int_values(data_info['unique_' + feature], cluster_results)
		write_encoding_to_file(feature_encoding, output_folder + feature + '_encoding' + '_' + level + '.csv')
	else:
		cluster_results = cluster_matrix_rows(context_matrix, 5, kmean_runs)
		feature_encoding = create_encoding_mapping_for_float_values(data[feature], data_info[feature + '_percentiles'], cluster_results)
		write_encoding_to_file(feature_encoding, output_folder + feature + '_encoding' + '_' + level + '.csv')

	return feature_encoding


# Example on how to use the encoding
# def main():
# 	bytes_encoding = encode(
# 		'_source_network_bytes',
# 		{'timestamp': '_source_@timestamp', 'src_ip':'_source_source_ip', 'dst_ip':'_source_destination_ip'},
# 		'conn',
# 		10,
# 		35,
# 		'./',
# 		'PATH/TO/DATA.csv',
# 	)

# if __name__ == '__main__':
# 	main()
	