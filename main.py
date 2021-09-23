import os.path

import numpy as np
from sklearn.preprocessing import normalize

# When the data format is bvecs or fvecs, the amount of data to be written
TOTAL_VECTOR_COUNT = 100000000
# Output chunk size (in number of vectors)
CHUNK_SIZE = 100000

QUERY_CHUNK_SIZE = 10000

# Does the data need to be normalized before insertion
IF_NORMALIZE = False


def load_npy_data(filename):
    data = np.load(filename)
    data = data.tolist()
    return data


def check_extracted_data():
    data_1B = load_npy_data("dataset-20M/vectors-0-99999.npy")
    print(data_1B[0])

    # This data is taken from https://github.com/milvus-io/bootcamp/blob/master/benchmark_test/lab1_sift1b_1m.md
    # Use to check if we extracted the 1B dataset correctly
    data_1M = load_npy_data("dataset-1M/binary_128d_00000.npy")
    print(data_1M[0])
    print(f"Both equal: {np.array_equal(data_1B, data_1M)}")


def create_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)


def load_ivecs_data(file_name):
    a = np.fromfile(file_name, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def ivecs_to_numpy(input_file_name, output_dir):
    data = load_ivecs_data(input_file_name)
    output_file_name = os.path.basename(input_file_name).replace(".ivecs", "")

    np.save(os.path.join(output_dir, output_file_name), data)


def load_bvecs_data(base_len, idx, file_name):
    """
    Taken from https://github.com/milvus-io/bootcamp/blob/5c1b1d414b9a1918a26c05d8ead1f3aeb8c318fc/benchmark_test/scripts/load.py#L39
    """
    begin_num = base_len * idx
    x = np.memmap(file_name, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    data = x.reshape(-1, d + 4)[begin_num:(begin_num + base_len), 4:]
    data = (data + 0.5) / 255
    if IF_NORMALIZE:
        data = normalize(data)
    data = data.tolist()
    return data


def bvecs_to_chunks(input_file_name, output_dir, file_name_prefix):
    chunks_count = 0
    vectors_count = 0

    while chunks_count < (TOTAL_VECTOR_COUNT // CHUNK_SIZE):
        chunk_start = vectors_count
        chunk_end = vectors_count + CHUNK_SIZE - 1
        print(f"Load chunk #{chunks_count + 1}: from {chunk_start} to {chunk_end}")
        chunk = load_bvecs_data(CHUNK_SIZE, chunks_count, input_file_name)

        np.save(os.path.join(output_dir, f"{file_name_prefix}-{chunk_start}-{chunk_end}"), chunk)
        chunks_count = chunks_count + 1
        vectors_count = vectors_count + len(chunk)


def bvecs_query_to_npy(input_file_name, output_dir, output_file_name):
    queries = load_bvecs_data(QUERY_CHUNK_SIZE, 0, input_file_name)

    np.save(os.path.join(output_dir, output_file_name), queries)


# --------------
# Convert ground truth data (list of k nearest neighbours indices)

# OUTPUT_DIR = "dataset-groundtruth"
# create_output_dir(OUTPUT_DIR)
# ivecs_to_numpy("dataset-1B/gnd/idx_1000M.ivecs", OUTPUT_DIR)
# data = load_npy_data("dataset-groundtruth/idx_1M.npy")
# print(data[0])

# --------------
# Extract chunks of data from 1B dataset
OUTPUT_DIR = "dataset-20M"
create_output_dir(OUTPUT_DIR)
bvecs_query_to_npy("dataset-1B/bigann_query.bvecs", OUTPUT_DIR, "queries")
bvecs_to_chunks("dataset-1B/bigann_base.bvecs", OUTPUT_DIR, "vectors")

# Uncomment if you want to check if data are extracted correctly
# check_extracted_data()

