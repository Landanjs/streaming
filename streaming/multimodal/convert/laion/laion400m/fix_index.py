from glob import glob
import json
import numpy as np
import os
from tqdm import tqdm


def func(idx):
    name = f'shard.{idx:05}.mds'
    local_path = f'/tmp/laion-shards/{name}'
    cmd = f'oci os object get -bn mosaicml-internal-dataset-laion --name {name} --file {local_path}'
    if os.system(cmd):
        raise RuntimeError(f'download failed: {cmd}')
    with open(local_path, 'rb') as f:
        b = f.read()
    n, z = map(int, np.frombuffer(b[:8], np.uint32))
    a = 4 + n * 4 + 4
    s = b[a:z].decode('utf-8')
    x = json.loads(s)
    hashes = {}
    x['hashes'] = hashes
    x['raw_data'] = {
    'basename': os.path.basename(local_path),
    'bytes': len(b),
    'hashes': hashes,
    }
    x['samples'] = n
    x['zip_data'] = None

    # Create index.json for shard
    index_dir = f'/tmp/laion-shards/{idx:05}.mds'
    os.mkdir(index_dir)
    index_path = os.path.join(index_dir, 'index.json')
    with open(index_path, 'w') as f:
        json.dump(x, f)
    os.remove(local_path)



from multiprocessing import Pool
if __name__ == '__main__':
    inds = list(range(29518))
    names = os.listdir('/tmp/laion-shards/')
    for name in names:
        num = int(name.split('.')[0])
        ind = inds.index(num)
        del inds[ind]
    print(len(inds))
    with Pool() as pool:
        pool.map(func, inds)