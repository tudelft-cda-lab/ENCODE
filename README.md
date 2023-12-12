# ENCODE: Encoding NetFlows for Network Anomaly Detection
[![Licence](https://img.shields.io/github/license/Ileriayo/markdown-badges?style=for-the-badge)](./LICENSE) ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

## About
A python package for learning contextual features from (semi-)continuous NetFlow feature data, with the aim to improve machine learning models for network anomaly detection. The underlying encoding algorithm is inspired by GloVe and it takes a distinct approach for computing contextual features from the input data:
- GloVe was designed to to learn vector representations for text. In contrast, ENCODE is designed to learn vector representations for NetFlow feature data.
- GloVE aggregates co-occurrences statistics of words computed from the input corpus, whereas ENCODE does no aggregation on the statistics computed from the input data. To be more specific, ENCODE keeps track of how often an arbitrary feature value *f'* occurs before and after another arbitrary feature value *f* in the input data. The before and after frequencies are store as separate columns in the co-occurrence matrix. Futhermore, ENCODE keeps track of how often *f* co-occurs with itself and the (logarithm) frequency of *f* in the input data.

The separation between before and after, and the self-occurrence statistics enables us to define a context in which the feature value *f* occurs and to differentiate between different NetFlow feature values. ENCODE then groups similare feature values by clustering the learned rows in the occurence matrix with the help of K-Means algorithms. As KMeans is a non-deterministic algorithm, ENCODE runs KMeans 10 times and selects the best clustering based on the Silhouette score.

## Installation
### Installing via PyPi
ENCODE can be easily install via PyPi by running the following command in your terminal:
```
pip install encode-netflow
``````

### Installing from source
It is also possile to install ENCODE from source. This can be done by first cloning this repository and then running the following command in the root directory of the repository:
```
pip install .
```

The `setup.py` is setup to install the required dependencies automatically when running the above command. However, it does not necessarilu install the same version of the dependencies as the ones listed in the `requirements.txt` file. If you want to install the exact same versions of the dependencies as the ones listed in the `requirements.txt` file, make sure you run the following command in the root directory of the repository before running the above command:
```
pip install -r requirements.txt
```

## Usage
To encode arbitrary (semi-)continuous NetFlow feature data, you need to first import the encode function from the encode module. Then, you can learn an encoding for your selected feature data by specifying the following parameters in the encode function (in the same exact order):
- `feature`: The name of the feature you want to encode. (e.g., 'bytes')
- `time_host_column_mapping`: A dictionary that maps the timestamp and host columns of the feature data to the corresponding column names. (e.g., `{'timestamp': 'timestamp', 'src_ip': 'src_ip', 'dst_ip': 'dst_ip'}`)
- `time_feature`: This is to specify whether the provided feature is a continuous feature or not (whether the data type is float or not). `True` should be specified if the feature is continuous and `False` otherwise.
-  `sorting_level`: How the feature data should be sorted to learn the encoding. Currently, ENCODE supports sorting on four levels: `conn` for sorting on the connection level, `src` for sorting on the source-host level, `ts` for sorting on the timestamp level, and `dst` for sorting on the destination-host level. By default, ENCODE sorts on the connection level.
- `kmeans_runs`: The number of times KMeans should be run to cluster the computed rows in the co-occurrence matrix. By default, ENCODE runs KMeans 10 times.
- `num_clusters`: The number of clusters KMeans should use to cluster the computed rows in the co-occurrence matrix. By default, ENCODE uses 35 clusters.
- `output_folder`: The folder where the outputs of ENCODE are stoerd. By default, ENCODE stores its output in the current working directory.
- `data_path`: The path to the NetFlow data file. Currently, ENCODE only support NetFlow data stored in CSV format. The NetFlow must contain at least the following features: the timestamps of the NetFlows, the source and destination IP addresses, and the feature you want to learn an encoding for. These column names must match the name given in the `time_host_column_mapping` and `feature` parameters. 

ENCODE also accepts a DataFrame as input instead of a path. In this case, you can ignore the `data_path` parameter and use the `data` parameter instead. Additionally, to save some computation time, you also use co-occurrence matrix computed from a previous run of ENCODE. To do so, you need to use the `precomputed_matrix=True` parameter and also specify where the precomputed matrix is stored using the `context_matrix_path` parameter. Do note that you should **only use** the `precomputed_matrix=True` parameter if you are using the same exact sample of NetFlow data as the previous run of ENCODE. Otherwise, the learned encoding might not be representative of the (new) NetFlow data.

The snippet below shows an example of how to use the `encode` function to learn an encoding for the `_source_network_bytes` feature:
```python
from encode.encode import encode

bytes_encoding = encode(
 		'_source_network_bytes',
 		{'timestamp': '_source_@timestamp', 'src_ip':'_source_source_ip', 'dst_ip':'_source_destination_ip'},
 		'conn',
 		10,
 		35, 		
        './',
		'PATH/TO/NETFLOW/DATA.csv',
 	)
```

The resulting encoding is stored as a dictionary, where the keys are the unique feature values and the values are the cluster assignments. 

## Citation
If you use ENCODE in your research, please cite the following paper:
```
@misc{cao2023encode,
      title={ENCODE: Encoding NetFlows for Network Anomaly Detection}, 
      author={Clinton Cao and Annibale Panichella and Sicco Verwer and Agathe Blaise and Filippo Rebecchi},
      year={2023},
      eprint={2207.03890},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Contact
If you have any questions, feel free to drop me an email. My email address is listed on GitHub page.


