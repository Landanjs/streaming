import os
from time import sleep
cache_dir = '/tmp/mds-cache/laion-filtered3'
done = False
shards = list(range(69000))
while not done:
    sleep(600)
    n_shards_curr = len(os.listdir(cache_dir))
    if n_shards_curr > 2000:
        print("DELETING SHARDS")
        for shard in shards[:7500]:
            os.remove(os.path.join(cache_dir, f'shard.{shard:05}.mds'))
        shards = shards[7500:]
        print('New shards:', shards)
    if len(shards) < 1000:
        done = True
    print("I sleep")
    sleep(600)
