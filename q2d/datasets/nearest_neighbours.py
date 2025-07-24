import logging
import os
import re
import time
from collections import defaultdict
from typing import Any, Dict, Optional, Union

from q2d.datasets.set_similarity_search.set_similarity_index import SetSimilarityIndex
from tqdm import tqdm

_ID_TYPE = Union[int, str]


class print_time:
    def __init__(self, desc):
        self.desc = desc

    def __enter__(self):
        print(self.desc)
        self.t = time.time()

    def __exit__(self, type, value, traceback):
        print(f"{self.desc} took {time.time()-self.t:.02f}s")


def verify_output(output_file: str) -> str:
    output_file = os.path.abspath(output_file)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if os.path.exists(output_file):
        raise RuntimeError(f"{output_file} OUTPUT FILE ALREADY EXISTS, CHANGE NAME!")
    return output_file


LOG = logging.getLogger(__name__)


def batch_generator(*args, batch_size: Optional[int] = None):
    """
    Generator that yields batches of samples from each input list. If batch_size is None,
    the entire list is returned as a single batch.

    :param *args: Variable length argument list, where each argument is a list of samples.
    Each list must have the same length.
    :param batch_size: The size of each batch to yield, or None to yield the full list.

    Yields: A tuple of lists, where each list contains a batch of samples from the corresponding input list.
    """
    if not all(len(lst) == len(args[0]) for lst in args):
        raise ValueError("All input lists must have the same length.")

    if batch_size is not None:
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if batch_size > len(args[0]):
            raise ValueError("batch_size must not exceed the length of the input lists.")
    else:
        # If batch_size is None, yield the entire lists as a single batch
        yield tuple(lst[:] for lst in args)
        return

    total_size = len(args[0])
    for start_idx in range(0, total_size, batch_size):
        end_idx = min(start_idx + batch_size, total_size)
        yield tuple(lst[start_idx:end_idx] for lst in args)


def find_similar_items(docs, func_ids, threshold: float = 0.9, batch_size: Optional[int] = None):
    """
    Find similar items among docs in all-to-all manner.
    :param docs: List of lists of strings.
    :param func_ids: List of function ids - each function id is related to list of strings.
    :param threshold: The Jaccard similarity threshold.
    :param batch_size: batch_size to use during querying the index.
    :return:
    """
    LOG.info("Finding exact matches...")
    doc2func_ids = defaultdict(list)
    for doc, func_id in tqdm(zip(docs, func_ids), desc="Building dict..."):
        doc2func_ids[doc].append(func_id)

    # find exact matches
    new_docs = []
    new_func_ids = []
    func_id2exact_matches_func_id = {}
    result_pairs = set()
    for doc, f_ids in tqdm(doc2func_ids.items(), desc="Compute exact matches..."):
        new_docs.append(doc)
        new_func_ids.append(f_ids[0])  # select only first element from exact matches
        if len(f_ids) > 1:
            func_id2exact_matches_func_id[f_ids[0]] = f_ids[1:]
            for pair in ((l, r) for l in f_ids for r in f_ids):
                result_pairs.add((*pair, 1))  # populate result with exact matches
    # near duplicates search
    with print_time("Building index for near duplicates"):
        index = SetSimilarityIndex(show_progress=True, similarity_threshold=threshold)
        index.build_index(index_dataset=new_docs, entities_names=new_func_ids)

    for batch_docs, batch_func_ids in tqdm(
        batch_generator(docs, func_ids, batch_size=batch_size), desc="searching near duplicates..."
    ):
        pairs = index.query_many(query_dataset=batch_docs, query_entities_names=batch_func_ids)
        for p in pairs:
            if p[0] != p[1]:
                result_pairs.add((p[0], p[1], p[2]))
                if p[0] in func_id2exact_matches_func_id:
                    for f_id in func_id2exact_matches_func_id[p[0]]:
                        if f_id != p[1]:
                            result_pairs.add((f_id, p[1], p[2]))
                if p[1] in func_id2exact_matches_func_id:
                    for f_id in func_id2exact_matches_func_id[p[1]]:
                        if f_id != p[0]:
                            result_pairs.add((f_id, p[0], p[2]))

    return result_pairs


def calculate_jaccard_similarities(
    dataset: Dict[_ID_TYPE, Dict[str, Any]],
    field: str = "doc",
    threshold: float = 0.9,
    batch_size: Optional[int] = None,
):
    """
    Calculate Jaccard similarity for documents in dataset using TF vectors, ignoring None values
    and maintaining a mapping from doc index to func_id.
    """

    # Filter out None documents, prepare inputs for SetSimilaritySearch
    def check_fields(content: Dict[str, Any], field: str) -> bool:
        splits = field.split("+")
        # 'doc' or 'code' cases
        if len(splits) == 1:
            return content[field] is not None and content[field].strip() != ""
        # 'doc+code' cases
        elif len(splits) == 2:
            return check_fields(content, splits[0]) and check_fields(content, splits[1])
        else:
            raise ValueError(f"Unexpected field {field}")

    def get_doc(content: Dict[str, Any], field: str) -> str:
        splits = field.split("+")
        # 'doc' or 'code' cases
        if len(splits) == 1:
            return content[field]
        # 'doc+code' cases
        elif len(splits) == 2:
            return content[splits[0]] + "\n" + content[splits[1]]
        else:
            raise ValueError(f"Unexpected field {field}")

    docs = []
    data_ids = []
    for data_id, content in tqdm(
        dataset.items(), total=len(dataset), desc="Preparing docs and func_ids..", leave=False
    ):
        if check_fields(content, field):
            curr_doc = tuple(
                s for s in re.split(r"[^a-zA-Z0-9]+", get_doc(content, field).lower()) if len(s.strip()) != 0
            )
            if curr_doc:
                docs.append(curr_doc)
                data_ids.append(data_id)
    print(f"{field}: Number of items to process: {len(docs)}")
    # Find similar docs in dataset
    result_pairs = find_similar_items(docs, data_ids, threshold=threshold, batch_size=batch_size)
    # Statistics
    print(f"{field}: Number of similar pairs: {len(result_pairs)}")
    print(f"{field}: Distribution of similarities scores:")
    print(
        f"{field}: Similarities scores equals to 1 in {sum(1 for s in result_pairs if s[2] == 1)} out of "
        f"{len(result_pairs)} pairs"
    )
    # Change to format {func_id: set([func_id1, func_id2, ...]),...}
    similar_items = defaultdict(set)
    for func_id1, func_id2, similarity in result_pairs:
        similar_items[func_id1].add((func_id2, similarity))
        similar_items[func_id2].add((func_id1, similarity))
    print(f"{field}: Number of items that have near duplicates {len(similar_items)} out of {len(docs)}")
    for func_id in similar_items:
        yield func_id, similar_items[func_id]


_NEAREST_SUFFIX = "_nearests"


def compute_near_duplicates(data: Dict[_ID_TYPE, Dict[str, Any]], field: str, batch_size: int, threshold: float):
    field_name = field + _NEAREST_SUFFIX
    start = time.time()
    for func_id, similar_func_ids in calculate_jaccard_similarities(
        data, field=field, threshold=threshold, batch_size=batch_size
    ):
        data[func_id][field_name] = list(similar_func_ids)
    end = time.time()
    duration_seconds = end - start
    hours, remainder = duration_seconds // 3600, duration_seconds % 3600
    minutes, seconds = remainder // 60, remainder % 60
    print(f"{field}: Pipeline duration: {int(hours)}:{int(minutes):02d}:{seconds:06.3f}")
