import os
from time import sleep
cache_dir = '/tmp/mds-cache/unfiltered-laion'
done = False
shards = list(range(41_409))
while not done:
    sleep(3600)
    n_shards_curr = len(os.listdir(cache_dir))
    if n_shards_curr > 10000:
        print("DELETING SHARDS")
        for shard in shards[:5000]:
            os.remove(os.path.join(cache_dir, f'shard.{shard:05}.mds'))
        shards = shards[5000:]
        print('New shards:', shards)
    print("I sleep")
