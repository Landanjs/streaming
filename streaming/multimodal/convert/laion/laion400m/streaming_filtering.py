from io import BytesIO

from tqdm import tqdm
import streaming
from torch.utils.data import DataLoader
from laion import StreamingLAIONDataset

columns = {
    'nsfw': 'str',
    'similarity': 'float64',
    'license': 'str',
    'caption': 'str',
    'url': 'str',
    'key': 'str',
    'status': 'str',
    'error_message': 'str',
    'width': 'int32',
    'height': 'int32',
    'original_width': 'int32',
    'original_height': 'int32',
    'exif': 'str',
    'jpg': 'bytes',
}

def noop_collate(batch):
    return batch
dataset = StreamingLAIONDataset(remote="oci://mosaicml-internal-dataset-laion400m-unfiltered2", local="/tmp/mds-cache/unfiltered-laion/")
dataloader = DataLoader(dataset, batch_size=512, num_workers=32, collate_fn=noop_collate)
size_limit = 512e6
count_shard = 0

with streaming.MDSWriter(out="oci://mosaicml-internal-dataset-laion/filtered2", columns=columns, compression=None, hashes=[], size_limit=256*(2**20), max_workers=32) as writer:
    for i, sample_batch in tqdm(enumerate(dataloader)):
        for sample in sample_batch:
            if sample['status'] == 'success':
                writer.write(sample)



