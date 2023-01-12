from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from collections import Counter
from .matrix_operations import compute_context_matrix, load_matrix
from .utils import *
from .frequency_utils import *

def process_data(data, level, columns_mapping):
	"""
	Precompute all the necessary data for creating the encoding.
	"""
	results = dict()
	print('Sorting NetFlows based on given level')
	data = data.sort_values(by=get_sorting_level(level, columns_mapping))

	print('Computing the unique bytes values of the dataset...')
	results['unique_bytes'] = sorted(data[columns_mapping['bytes']].unique())
	results['unique_bytes_frequency'] = Counter(data[columns_mapping['bytes']])

	print('Computing the unique packets values of the dataset...')
	results['unique_packets'] = sorted(data[columns_mapping['packets']].unique())
	results['unique_packets_frequency'] = Counter(data[columns_mapping['packets']])

	print('Computing the unique_durations...')
	results['unique_duration'] = sorted(data[columns_mapping['duration']].unique())

	print('Computing the percentiles of durations...')
	results['duration_percentiles'] = compute_percentiles(data[columns_mapping['duration']], 5, 105, 5)

	print('Computing frequency of the duration percentiles...')
	raw_duration_to_percentiles = []
	for d in data[columns_mapping['duration']]:
		raw_duration_to_percentiles.append(find_percentile(d, results['duration_percentiles']))

	results['duration_percentiles_frequency'] = Counter(raw_duration_to_percentiles)

	column_index_mapping = find_column_index_mapping(columns_mapping, data.columns.tolist())
	
	print('Computing the counts of the next and previous symbols...')
	data_np = data.to_numpy()
	next_byte_counts, next_byte_symbols = compute_next_counts(data_np, 'bytes', level, column_index_mapping)
	results['next_byte_percentiles'] = compute_percentiles(next_byte_symbols, 10, 110, 10)
	results['next_byte_counts'] = convert_symbols_to_percentiles(next_byte_counts, results['next_byte_percentiles'])
	previous_byte_count, previous_byte_symbols = compute_previous_counts(data_np, 'bytes', level, column_index_mapping)
	results['previous_byte_percentiles'] = compute_percentiles(previous_byte_symbols, 10, 110, 10)
	results['previous_byte_counts'] = convert_symbols_to_percentiles(previous_byte_count, results['previous_byte_percentiles'])

	next_packet_counts, next_packet_symbols = compute_next_counts(data_np, 'packets', level, column_index_mapping)
	results['next_packet_percentiles'] = compute_percentiles(next_packet_symbols, 10, 110, 10)
	results['next_packet_counts'] = convert_symbols_to_percentiles(next_packet_counts, results['next_packet_percentiles'])
	previous_packet_count, previous_packet_symbols = compute_previous_counts(data_np, 'packets', level, column_index_mapping)
	results['previous_packet_percentiles'] = compute_percentiles(previous_packet_symbols, 10, 110, 10)
	results['previous_packet_counts'] = convert_symbols_to_percentiles(previous_packet_count, results['previous_packet_percentiles'])

	next_dur_counts, next_dur_symbols = compute_next_counts_timed(data_np, 'duration', level, 500, column_index_mapping)
	results['next_dur_percentiles'] = compute_percentiles(next_dur_symbols, 10, 110, 10)
	results['next_dur_counts'] = convert_symbols_to_percentiles_timed(next_dur_counts, results['duration_percentiles'], results['next_dur_percentiles'])
	previous_dur_counts, previous_dur_symbols = compute_previous_counts_timed(data_np, 'duration', level, 500, column_index_mapping)
	results['previous_dur_percentiles'] = compute_percentiles(previous_dur_symbols, 10, 110, 10)
	results['previous_dur_counts'] = convert_symbols_to_percentiles_timed(previous_dur_counts, results['duration_percentiles'], results['previous_dur_percentiles'])

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

	return percentile_to_cluster_mapping

def encode(columns_mapping, level='conn', kmean_runs=10, num_clusters=35, output_folder='./', data_path=None, data=None, precomputed_matrix=False, bytes_context_matrix_path = None, packets_context_matrix_path = None, durations_context_matrix_path = None):
	"""
	Encoded the given NetFlow data.
	"""
	print('Reading and processing data...')
	if data_path is not None:
		data =  read_csv_file(data_path, columns_mapping.values())
	else:
		data = data
	
	data_info = process_data(data, level, columns_mapping)

	if precomputed_matrix:
		if bytes_context_matrix_path is None or packets_context_matrix_path is None or durations_context_matrix_path is None:
			raise ValueError('Must provide paths to the precomputed matrices if precomputed_matrix is set to True')

		print('Using precomputed matrix...')
		bytes_context_matrix, _ = load_matrix(bytes_context_matrix_path)
		packets_context_matrix, _ = load_matrix(packets_context_matrix_path)
		durations_context_matrix, _ = load_matrix(durations_context_matrix_path)
	
	else:
		print('Computing the matrices...')
		bytes_context_matrix = compute_matrix(
			data_info['unique_bytes'],
			data_info['next_byte_percentiles'],
			data_info['previous_byte_percentiles'],
			data_info['next_byte_counts'],
			data_info['previous_byte_counts'],
			data_info['unique_bytes_frequency'],
			output_folder + 'bytes_context_matrix_' + level + '.csv',
			)

		packets_context_matrix = compute_matrix(
			data_info['unique_packets'],
			data_info['next_packet_percentiles'],
			data_info['previous_packet_percentiles'],
			data_info['next_packet_counts'],
			data_info['previous_packet_counts'],
			data_info['unique_packets_frequency'],
			output_folder + 'packets_context_matrix_' + level + '.csv',
			)
		
		durations_context_matrix = compute_matrix(
			[x for x in range(20)],
			data_info['next_dur_percentiles'],
			data_info['previous_dur_percentiles'],
			data_info['next_dur_counts'],
			data_info['previous_dur_counts'],
			data_info['duration_percentiles_frequency'],
			output_folder + 'durations_context_matrix_' + level + '.csv',
			)

	bytes_cluster_results = cluster_matrix_rows(bytes_context_matrix, num_clusters, kmean_runs)
	bytes_encoding = create_encoding_mapping_for_int_values(data_info['unique_bytes'], bytes_cluster_results)
	write_encoding_to_file(bytes_encoding, output_folder + 'bytes_encoding' + '_' + level + '.csv')
	packets_cluster_results = cluster_matrix_rows(packets_context_matrix, num_clusters, kmean_runs)
	packets_encoding = create_encoding_mapping_for_int_values(data_info['unique_packets'], packets_cluster_results)
	write_encoding_to_file(packets_encoding, output_folder + 'packets_encoding' + '_' + level + '.csv')
	duration_cluster_results = cluster_matrix_rows(durations_context_matrix, 5, kmean_runs)
	duration_encoding = create_encoding_mapping_for_float_values(data[columns_mapping['duration']], data_info['duration_percentiles'], duration_cluster_results)
	write_encoding_to_file(duration_encoding, output_folder + 'duration_encoding' + '_' + level + '.csv')

	return bytes_encoding, packets_encoding, duration_encoding


# Example on how to use the encoding
# def main():
# 	bytes_encoding, packets_encoding, duration_encoding = encode(
# 		'elastic_data.csv',
# 		{'timestamp': '_source_@timestamp', 'src_ip':'_source_source_ip', 'dst_ip':'_source_destination_ip', 'bytes':'_source_network_bytes', 'packets':'_source_network_packets', 'duration':'_source_event_duration'},
# 		'conn',
# 		10,
# 		35,
# 		'./'
# 	)

# if __name__ == '__main__':
# 	main()
	