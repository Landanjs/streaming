import os
import json

index = []
for i in range(41408):
    obj = json.load(open(f'/tmp/laion-shards/{i:05}.mds/index.json'))
    index.append(obj)

obj2 = {'version': 2, 'shards': index}
json.dump(obj2, open('/tmp/laion-shards/index.json', 'w'))
#os.system('oci os object put -bn mosaicml-internal-dataset-laion --file /tmp/laion-shards/index.json --name index.json')