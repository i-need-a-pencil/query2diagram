import logging
import os
import sys
from typing import Any, Hashable, Iterable, List, Sequence, Set, Tuple, Union

import numpy as np
import psutil
from q2d.datasets.set_similarity_search import utils

LOG = logging.getLogger(__name__)


class SetSimilarityIndex:
    """
    This data structure supports set similarity search queries and all pairs searching. The algorithm is based on a
    combination of the prefix filter and position filter techniques
    """

    INDEX_MMAP_FILENAME = "index_mmap"

    def __init__(
        self,
        similarity_func_name: str = "jaccard",
        similarity_threshold: float = 0.9,
        show_progress: bool = False,
        n_cores: int = os.cpu_count(),
    ):
        """
        :param similarity_func_name: the name of the similarity function used this function currently supports
        `"jaccard"`, `"cosine"`, `"containment"`, `"containment_min"`
        :param similarity_threshold: the threshold used, must be in (0, 1]
        :param show_progress: flag to use `tqdm` progress bar or not
        :param n_cores: number of cores to use while processing many queries to index
        """
        if similarity_func_name not in utils.similarity_funcs:
            raise ValueError("Similarity function {} is not supported".format(similarity_func_name))
        if not isinstance(similarity_threshold, float):
            raise TypeError("Similarity threshold must be float value")
        if not 0 < similarity_threshold <= 1.0:
            raise ValueError("Similarity threshold must be in the range (0, 1]")
        self.similarity_func_name = similarity_func_name
        self.similarity_threshold = similarity_threshold
        self.index_data = None
        self.index_pointers = None
        self.index = None
        self.token2id = None
        self.token_index_size = None
        self.token2int_value = None
        self.entities_names = None
        self.show_progress = show_progress
        self.n_cores = n_cores

    def _encode(self, dataset: Iterable[Sequence[Hashable]]) -> Iterable[np.ndarray]:
        """
        Transform a dataset to a numeric format - assign index to each unique value and replace input values with index
        :param dataset: iterable of feature vectors like ["a", "b", "c"], ["d", "b"], ...
        :return: generator of integer_dataset - [0, 1, 2], [3, 1], ...
        """
        for features in dataset:
            yield np.array(
                [self.token2int_value.setdefault(token, len(self.token2int_value)) for token in features],
                dtype=np.int32,
            )

    def save_index(self, storage_directory: str) -> None:
        """
        Save index to directory
        :param storage_directory: path to folder where index will be saved
        """
        os.makedirs(storage_directory, exist_ok=True)
        np.save(os.path.join(storage_directory, "index_data"), self.index_data)
        np.save(os.path.join(storage_directory, "index_pointers"), self.index_pointers)
        np.save(os.path.join(storage_directory, "token2id"), self.token2id)
        np.save(os.path.join(storage_directory, "index"), self.index)
        np.save(os.path.join(storage_directory, "token_index_size"), self.token_index_size)
        np.save(os.path.join(storage_directory, "entities_names"), self.entities_names)
        np.save(os.path.join(storage_directory, "token2int_value"), self.token2int_value)

    def load_index(self, storage_directory: str, mmap: bool = False) -> None:
        """
        Load index from storage directory at disk
        :param storage_directory: path to folder where index is stored
        :param mmap: if mmap loaded index and indexed_data or not
        """
        self.index_data = np.load(os.path.join(storage_directory, "index_data.npy"), mmap_mode="r" if mmap else None)
        self.index_pointers = np.load(os.path.join(storage_directory, "index_pointers.npy"))
        self.index = np.load(os.path.join(storage_directory, "index.npy"), mmap_mode="r" if mmap else None)
        self.token2id = np.load(os.path.join(storage_directory, "token2id.npy"))
        self.token_index_size = np.load(os.path.join(storage_directory, "token_index_size.npy"))
        self.entities_names = np.load(os.path.join(storage_directory, "entities_names.npy"))
        self.token2int_value = np.load(os.path.join(storage_directory, "token2int_value.npy"), allow_pickle=True).item()

    def _init_index_structures(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Init index structures for index building step
        :return: token_index_size - empty matrix with start and end positions for each token in index
                 current_token_pos - empty array which will maintain current position for each token in index
                 index_prefix_sizes - size of prefix for each feature vector
                 index_size - size of index
        """
        n_tokens = len(self.token2id)
        token_index_size = np.zeros((n_tokens, 2), dtype=np.int32)
        current_token_pos = np.zeros((n_tokens,), dtype=np.int32)
        index_prefix_sizes = np.zeros((self.index_pointers.shape[0],), dtype=np.int32)
        index_size = utils.calculate_index_size_funcs[self.similarity_func_name](
            pointers=self.index_pointers,
            prefix_sizes=index_prefix_sizes,
            similarity_threshold=self.similarity_threshold,
        )
        return token_index_size, current_token_pos, index_prefix_sizes, index_size

    def build_index(
        self,
        index_dataset: Iterable[Sequence[Hashable]],
        features_lengths: np.ndarray = None,
        entities_names: Sequence[Any] = None,
        mmap: bool = False,
    ) -> None:
        """
        Build index for dataset
        :param index_dataset: iterable of vector features
        :param features_lengths: array of lengths for each feature vector from index_dataset
        :param entities_names: sequence of entities names for each feature vector
        :param mmap: if mmap index or not
        """
        LOG.debug("Starting: building index step")
        process = psutil.Process(os.getpid())
        LOG.debug(f"Process taken memory beginning build index MB {process.memory_info().rss / 1024**2}")

        LOG.debug("Starting: checking features_lengths")
        if features_lengths is None:
            if isinstance(index_dataset, Sequence):
                features_lengths = np.array([len(features) for features in index_dataset], dtype=np.int32)
            else:
                raise ValueError("features_lengths argument should be provided")
        LOG.debug("Finished: checking features_lengths")

        LOG.debug("Starting: checking entities_names")
        if entities_names is None:
            entities_names = np.arange(len(features_lengths))
        self.entities_names = entities_names
        LOG.debug("Finished: checking entities_names")

        LOG.debug("Starting: encoding step")
        self.token2int_value = dict()
        integer_dataset_generator = self._encode(index_dataset)
        LOG.debug("Finished: encoding step")

        LOG.debug("Starting: frequency order transformation step")
        self.index_data, self.index_pointers, self.token2id = utils.frequency_order_transform(
            index_dataset=integer_dataset_generator,
            features_lengths=features_lengths,
            mmap=mmap,
        )
        LOG.debug("Finished: frequency order transformation step")
        LOG.debug(f"Taken memory after freq_order_trans MB {process.memory_info().rss / 1024 ** 2}")

        LOG.debug("Starting: init index structures step")
        self.token_index_size, current_token_pos, index_prefix_sizes, index_size = self._init_index_structures()
        LOG.debug("Finished: init index structures step")

        LOG.debug("Starting: fill index step")
        if mmap:
            self.index = np.memmap(filename=self.INDEX_MMAP_FILENAME, dtype=np.int32, mode="w+", shape=(index_size, 2))
        else:
            self.index = np.zeros((index_size, 2), dtype=np.int32)
        utils.fill_index(
            index=self.index,
            token_index_size=self.token_index_size,
            current_token_pos=current_token_pos,
            input_pointer=self.index_pointers,
            input_data=self.index_data,
            prefix_sizes=index_prefix_sizes,
        )
        LOG.debug("Finished: fill index step")

        LOG.debug(
            f"Index size MB {self.index.nbytes / 1024**2}; \n"
            f"Token_index_size size MB {self.token_index_size.nbytes / 1024**2}; \n"
            f"Index_prefix_size MB {index_prefix_sizes.nbytes / 1024**2}; \n"
            f"Current_token_pos MB {current_token_pos.nbytes / 1024**2}; \n"
            f"Token2int_value size MB {sys.getsizeof(self.token2int_value) / 1024 ** 2}; \n"
            f"Finished building index size MB {process.memory_info().rss / 1024**2}"
        )
        LOG.debug("Finished: building index step")

    def _encode_queries(self, query_dataset: Sequence[Sequence[Hashable]]) -> List[np.ndarray]:
        """
        Transform queries to a numeric format - assign index to each unique value and replace input values with index
        :param query_dataset: sequence of vector features will be in query
        :return: numeric queries dataset
        """
        encoded_queries = []
        unknown_token_int_value = len(self.token2int_value)
        for query_features in query_dataset:
            numeric_query_features = []
            for token in query_features:
                if token in self.token2int_value:
                    numeric_query_features.append(self.token2int_value[token])
                else:
                    numeric_query_features.append(unknown_token_int_value)
                    unknown_token_int_value += 1
            encoded_queries.append(np.array(numeric_query_features))
        return encoded_queries

    def query(self, features: Sequence[Hashable]) -> Set[Tuple[Union[Any, int], float]]:
        """
        Query features to index
        :param features: feature vector
        :return: list of tuples `(candidate_entity_name/candidate_index, similarity_score)` for truly similar vectors
        """
        query_features = self._encode_queries([features])
        query_data, _ = utils.frequency_order_transform_queries(queries_dataset=query_features, token2id=self.token2id)
        result_set = utils.query(
            similarity_func_name=self.similarity_func_name,
            features=query_data,
            index=self.index,
            token_index_size=self.token_index_size,
            index_data=self.index_data,
            index_pointers=self.index_pointers,
            similarity_threshold=self.similarity_threshold,
        )
        return {
            (self.entities_names[candidate_index], similarity_score) for candidate_index, similarity_score in result_set
        }

    def query_many(
        self,
        query_dataset: Sequence[Sequence[Hashable]],
        query_entities_names: Sequence[Any] = None,
    ) -> Set[Tuple[Union[Any, int], Union[Any, int], float]]:
        """
        Query batch of vector features to index
        :param query_dataset: sequence of vector features will be in query
        :param query_entities_names: sequence of entities names for each query vector
        :return: set of tuples `(query_entity_name/index_from_query_dataset,
                                 indexed_entity_name/index_from_indexed_dataset,
                                 similarity_score
                                )` for truly similar feature vectors
        """
        if query_entities_names is None:
            query_entities_names = list(range(len(query_dataset)))
        numeric_query_features_dataset = self._encode_queries(query_dataset)
        query_data, query_pointers = utils.frequency_order_transform_queries(
            numeric_query_features_dataset, self.token2id
        )
        result = utils.query_batch(
            similarity_func_name=self.similarity_func_name,
            query_data=query_data,
            query_pointers=query_pointers,
            index=self.index,
            token_index_size=self.token_index_size,
            index_data=self.index_data,
            index_pointers=self.index_pointers,
            similarity_threshold=self.similarity_threshold,
            show_progress=self.show_progress,
            n_cores=self.n_cores,
        )
        return {
            (query_entities_names[index_from_query_data], self.entities_names[index_from_index_data], similarity_score)
            for index_from_query_data, index_from_index_data, similarity_score in result
        }

    def all_pairs(
        self,
        dataset: Sequence[Sequence[Hashable]],
        entities_names: Sequence[Any] = None,
    ) -> Set[Tuple[int, int, float]]:
        """
        Find all pairs of similar vectors in dataset with similarity higher that self.similarity_threshold
        :param dataset: sequence of feature vectors
        :param entities_names: sequence of entities names for each feature vecto
        :return: set of tuples `(index_from_dataset, index_from_dataset, similarity_score)` for truly similar feature
                 vectors
        """
        self.build_index(index_dataset=dataset, entities_names=entities_names)
        pairs = self.query_many(query_dataset=dataset, query_entities_names=entities_names)
        is_symmetric_similarity_function = self.similarity_func_name in utils.symmetric_similarity_funcs

        result_pairs = set()
        for p in pairs:
            if p[0] != p[1]:
                if is_symmetric_similarity_function:
                    result_pairs.add((*sorted([p[0], p[1]], reverse=True), p[2]))
                else:
                    result_pairs.add(p)
        return result_pairs
