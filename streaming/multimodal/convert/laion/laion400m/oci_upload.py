import os
import multiprocessing
import json
from time import sleep

def multi_proc(name):
    ind, _ = name.split('.')
    shard_name = os.path.join(base_folder, f'{ind}.mds', 'shard.00000.mds')
    cmd = f'oci os object put -ns axhe5a72vzpp -bn mosaicml-internal-dataset-laion400m-filtered --file {shard_name} --name shard.{ind}.mds'
    if os.system(cmd):
        raise RuntimeError(f'Download failed')
    os.remove(shard_name)
    os.rename(f'{base_folder}{ind}.done', f'{base_folder}{ind}.donedone')
    print(f'Shard {ind}: done')


done = False
base_folder = '/tmp/laion-filtered/'
while not done:
    basenames = set(os.listdir(base_folder))
    filtered_names = list(filter(lambda s: s.endswith('.done'), basenames))
    with multiprocessing.Pool(8) as pool:
        pool.map(multi_proc, filtered_names)
    sleep(15)
    if os.path.exists('/tmp/laion-filtered/done'):
        done = True

# create index
infos = []
basenames = set(os.listdir(base_folder))
filtered_names = list(filter(lambda s: s.endswith('.donedone'), basenames))
for name in filtered_names:
    ind, _ = name.split('.')
    sub_index_filename = os.path.join(base_folder, f'{ind}.mds', 'index.json')
    obj = json.load(open(sub_index_filename))
    info, = obj['shards']
    infos.append(info)

obj = {
    'version': 2,
    'shards': infos,
}

local = os.path.join(base_folder, 'index.json')
with open(local, 'w') as out:
    json.dump(obj, out)

cmd = f'oci os object put -ns axhe5a72vzpp -bn mosaicml-internal-dataset-laion400m-filtered --file {local} --name index.json'
if os.system(cmd):
    raise RuntimeError(f'failed to upload index.json')

