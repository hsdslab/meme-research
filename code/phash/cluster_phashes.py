'''
This code is part of the publication "On the Origins of Memes by Means of Fringe Web Communities" at IMC 2018.
If you use this code please cite the publication.
'''
import matplotlib.image as mpimg
import json
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pickle
from optparse import OptionParser
import os
from tqdm import tqdm
from sklearn.cluster import HDBSCAN
from datetime import datetime

def load_index_image_mapping(index_image_file):
    return pickle.load(open(index_image_file, 'rb'))

def load_phashes(fil):
    phash_dict = {}
    with open(fil) as f:
        for line in f:
            el = line.replace('\n','').split('\t')
            image = el[0]
            phash = el[1]
            phash_dict[image] = phash
    return phash_dict

def find_cluster_medroid_phash(cl_output, cluster_num):
    medoids = cl_output.medoids_
    medoid = medoids[cluster_num]
    medoid_idx = np.where((X == medoid).all(axis=1))[0][0]
    medoid_image = index_image[medoid_idx]
    medoid_phash = image_to_phash[medoid_image]
    return medoid_phash, medoid_image

def print_clusters_to_file(clustering_output, filename):
    num_clusters = len(dict(Counter(clustering_output.labels_)).keys())
    clusters = clustering_output.labels_.tolist()
    output = open(filename, 'w')
    #output_json = {}
    for k in range(-1, num_clusters):
        output_json= {}
        indices = [i for i, x in enumerate(clusters) if x == k]
        #output.write( "Cluster = %d\n" %k)
        images = []
        if k % 100 == 0:
            print("Calculating medoids. Cluster: %d/%d" %(k, num_clusters))

        if len(indices) > 0:
            for j in indices:
                image = index_image[j]
                images.append(image)
            output_json['cluster_no'] = k
            output_json['images'] = images
            if k!=-1:
                medroid, medroid_path = find_cluster_medroid_phash(clustering_output, k)
                output_json['medroid_phash'] = medroid
                output_json['medroid_path'] = medroid_path
                output.write(json.dumps(output_json) + '\n')
    output.close()
    return output_json

def hex_to_binary(hex_string):
    binary_string = bin(int(hex_string, 16))[2:].zfill(64)
    return binary_string

def binary_to_hex(binary_array: np.ndarray):
    binary_array = binary_array.astype(int)
    binary_string = "".join(binary_array.astype(str))
    hex_string = hex(int(binary_string, 2))[2:]
    return hex_string

def read_file(file_path) -> list:
    path_hash_pairs = []
    with open(file_path, 'r') as file:
        for line in file.readlines():
            path, hash_code = line.strip().split('\t')
            path_hash_pairs.append((path, hash_code))
    print('[i] processed', len(path_hash_pairs))
    return path_hash_pairs
    
def read_phashes_manifest(phashes_path):
    phashes = {}
    with open(phashes_path) as infile:
        for line in infile.readlines():
            split = line.split('\t')
            hashid = split[0].strip()
            hash_str = split[1].strip()
            phashes[hashid] = hash_str
    print('[i] processed', len(phashes))
    return phashes

def hex_to_hash(hexstr, hash_size=8):

    l = []
    count = hash_size * (hash_size // 4)
    if len(hexstr) != count:
        emsg = 'Expected hex string size of {}.'
        raise ValueError(emsg.format(count))
    for i in range(count // 2):
        h = hexstr[i*2:i*2+2]
        v = int("0x" + h, 16)
        l.append([v & 2**i > 0 for i in range(8)])
    return np.array(l).flatten()#.astype(int)
    
def precompute_vectors(hashes, phases_path):
    pickle_file = phases_path + '.pickle'
    if os.path.isfile(pickle_file): 
        with open(pickle_file, 'rb') as fo:
            hashes = pickle.load(fo)
        print('[w] fetch precomputed vectors from ', pickle_file, 'new processed', len(hashes))
        return np.array(hashes)
    else:
        hashes = np.array(list(hashes.values()))
        hashes2 = []
        for hex_hash in hashes:
            try:
                hashes2.append(hex_to_hash(hex_hash))
            except Exception as e:
                print(hex_hash)
                print(str(e))
        with open(pickle_file, 'wb') as fo:
            pickle.dump(hashes2, fo)
    return np.array(hashes2)

def create_index_image_mapping(phashes):
    with open(phashes, 'r') as f:
        phash_list = f.readlines()
        index_image = {}

        for i, line in enumerate(phash_list):
            index_image[i] = line.strip().split('\t')[0]

    with open('./index_image.p','wb') as f:
        pickle.dump(index_image, f) 

    return index_image


if __name__ == "__main__":


    # load data to dictionary
    parser = OptionParser()
    parser.add_option("-p", "--phashes", dest='phashes', default='phashes.txt', help="file with phashes")
    parser.add_option("-i", "--index", dest='index', default='index_image.p',help="dictionary that includes the mapping between images and index in distance matrix")
    parser.add_option("-o", "--output", dest='output', default='clustering_output.txt',help="file that includes clustering output")
    parser.add_option("-t", "--threshold", dest='threshold', default=8, help="threshold for clustering")
    parser.add_option("-m", "--min_samples", dest='min_samples', default=5, help="min_samples for clustering")

    (options, args) = parser.parse_args()

    phashes = options.phashes
    clustering_output = options.output
    index_image_file = options.index

    CLUSTERING_THRESHOLD = options.threshold
    CLUSTERING_MIN_SAMPLES = options.min_samples

    start_time = datetime.now()
    print("Script started at: ", start_time.strftime("%Y-%m-%d %H:%M:%S"))

    index_image = create_index_image_mapping(phashes)

    image_to_phash = read_phashes_manifest(phashes)
    X = precompute_vectors(image_to_phash, phashes)

    print("Shape of X: ", X.shape)

    print("Clustering...")
    clustering = HDBSCAN(metric='hamming',store_centers='medoid', n_jobs=-1, min_samples=CLUSTERING_MIN_SAMPLES).fit(X)
    try:
        with open('./clustering_output.pickle', 'wb') as fo:
            pickle.dump(clustering, fo)
    except:
        with open('clustering_output.pickle', 'wb') as fo:
            pickle.dump(clustering, fo)
    print("Clustering done")
    num_clusters = len(dict(Counter(clustering.labels_)).keys())
    print("Number of clusters  = %d " %(num_clusters-1))
        
    print("Calculating cluster medoids...")
    output_json = print_clusters_to_file(clustering, clustering_output)
    print("Clustering output written to %s" %(clustering_output))
    print("Script finished at: ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("Time taken: ", datetime.now() - start_time)
