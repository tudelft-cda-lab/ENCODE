from .utils import check_level, find_percentile

def convert_symbols_to_percentiles(symbols_count, percentiles):
	"""
	For each symbol, find its corresponding percentile and increment the 
	frequency of that corresponding bin/percentile.
	"""
	converted_counts = dict()
	for s in symbols_count:
		converted_counts[s] = dict()
		for i in range(len(percentiles)):
			converted_counts[s][i] = 0.0
		for e in symbols_count[s]:
			if e == 'N' or e == 'SELF':
				converted_counts[s][e] = symbols_count[s][e]
			else:
				value_percentile = find_percentile(e, percentiles)
				converted_counts[s][value_percentile] += symbols_count[s][e]
	
	return converted_counts


def convert_symbols_to_percentiles_timed(timed_symbols_count, percentiles, timed_count_percentiles):
	"""
	This function is same as convert_symbols_to_percentiles, except that this is used 
	for time features (e.g. duration).
	"""
	converted_counts = dict()

	for i in range(len(percentiles)):
		converted_counts[i] = dict()
		converted_counts[i]['N'] = 0.0
		converted_counts[i]['SELF'] = 0.0
	
	for s in timed_symbols_count:
		percentile = find_percentile(s, percentiles)
		converted_counts[percentile] = dict()
		
		for i in range(len(timed_count_percentiles)):
			converted_counts[percentile][i] = 0.0
		
		converted_counts[percentile]['N'] = 0.0
		converted_counts[percentile]['SELF'] = 0.0
		
		for e in timed_symbols_count[s]:
			if e == 'N' or e == 'SELF':
				converted_counts[percentile][e] += timed_symbols_count[s][e]
			else:
				value_percentile = find_percentile(e, timed_count_percentiles)
				converted_counts[percentile][value_percentile] += timed_symbols_count[s][e]

	return converted_counts


def compute_next_counts(feature_data, feature, level, column_index_mapping):
	"""
	For each given symbol, compute the frequency of the corresponding next symbol
	and store this in a dictionary for lookup.
	"""
	next_counts = dict()
	next_symbols = []
	for i in range(len(feature_data)):
		row = feature_data[i]
		value = row[column_index_mapping[feature]]

		if value not in next_counts:
			next_counts[value] = dict()
			next_counts[value]['N'] = 0.0
			next_counts[value]['SELF'] = 0.0

		if i + 1 < len(feature_data):
			next_row = feature_data[i+1]
			if check_level(level, row, next_row, column_index_mapping):
				next_symbols.append(next_row[column_index_mapping[feature]])
				if value == next_row[column_index_mapping[feature]]:
					next_counts[value]['SELF'] += 1.0
				else:
					if next_row[column_index_mapping[feature]] not in next_counts[value]:
						next_counts[value][next_row[column_index_mapping[feature]]] = 0.0

					next_counts[value][next_row[column_index_mapping[feature]]] += 1.0
			else:
				next_counts[value]['N'] += 1.0
		else:
			next_counts[value]['N'] += 1.0
			break;
	
	return next_counts, next_symbols


def compute_previous_counts(feature_data, feature, level, column_index_mapping):
	"""
	This function is same as compute_next_counts, except that this is used to 
	compute the frequency of the previous symbol.
	"""
	previous_counts = dict()
	previous_symbols = []
	for i in range(len(feature_data)):
		row = feature_data[i]
		value = row[column_index_mapping[feature]]

		if value not in previous_counts:
			previous_counts[value] = dict()
			previous_counts[value]['N'] = 0.0
			previous_counts[value]['SELF'] = 0.0

		if i == 0:
			previous_counts[value]['N'] += 1.0
			continue
		
		previous_row = feature_data[i-1]
		if check_level(level, row, previous_row, column_index_mapping):
			previous_symbols.append(previous_row[column_index_mapping[feature]])
			if value == previous_row[column_index_mapping[feature]]:
				previous_counts[value]['SELF'] += 1.0
			else:
				if previous_row[column_index_mapping[feature]] not in previous_counts[value]:
					previous_counts[value][previous_row[column_index_mapping[feature]]] = 0.0

				previous_counts[value][previous_row[column_index_mapping[feature]]] += 1.0
		else:
			previous_counts[value]['N'] += 1.0
	
	return previous_counts, previous_symbols


def compute_next_counts_timed(timed_feature_data, feature, level, threshold, column_index_mapping):
	"""
	This function is same as compute_next_counts, except that this is used 
	for time features (e.g. duration).
	"""
	next_counts = dict()
	next_symbols = []

	for i in range(len(timed_feature_data)):
		row = timed_feature_data[i]
		value = row[column_index_mapping[feature]]

		if value not in next_counts:
			next_counts[value] = dict()
			next_counts[value]['N'] = 0.0
			next_counts[value]['SELF'] = 0.0

		if i + 1 < len(timed_feature_data):
			next_row = timed_feature_data[i+1]
			if check_level(level, row, next_row, column_index_mapping):
				next_symbols.append(next_row[column_index_mapping[feature]])
				if abs(value - next_row[column_index_mapping[feature]]) <= threshold:
					next_counts[value]['SELF'] += 1.0
				else:
					if next_row[column_index_mapping[feature]] not in next_counts[value]:
						next_counts[value][next_row[column_index_mapping[feature]]] = 0.0

					next_counts[value][next_row[column_index_mapping[feature]]] += 1.0
			else:
				next_counts[value]['N'] += 1.0
		else:
			next_counts[value]['N'] += 1.0
			break;
	
	return next_counts, next_symbols


def compute_previous_counts_timed(timed_feature_data, feature, level, threshold, column_index_mapping):
	"""
	This function is almost same as compute_previous_counts, except that this is
	for time features (e.g. duration).
	"""
	previous_counts = dict()
	previous_symbols = []

	for i in range(len(timed_feature_data)):
		row = timed_feature_data[i]
		value = row[column_index_mapping[feature]]

		if value not in previous_counts:
			previous_counts[value] = dict()
			previous_counts[value]['N'] = 0.0
			previous_counts[value]['SELF'] = 0.0

		if i == 0:
			previous_counts[value]['N'] += 1.0
			continue
		
		previous_row = timed_feature_data[i-1]
		if check_level(level, row, previous_row, column_index_mapping):
			previous_symbols.append(previous_row[column_index_mapping[feature]])
			if abs(value - previous_row[column_index_mapping[feature]]) <= threshold:
				previous_counts[value]['SELF'] += 1.0
			else:
				if previous_row[column_index_mapping[feature]] not in previous_counts[value]:
						previous_counts[value][previous_row[column_index_mapping[feature]]] = 0.0

				previous_counts[value][previous_row[column_index_mapping[feature]]] += 1.0
		else:
			previous_counts[value]['N'] += 1.0
	
	return previous_counts, previous_symbols