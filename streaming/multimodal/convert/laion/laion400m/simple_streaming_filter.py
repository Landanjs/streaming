from typing import Optional
from tqdm import tqdm
import streaming
from torch.utils.data import DataLoader


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


class StreamingLAIONDataset(streaming.StreamingDataset):
    """
    Implementation of the LAION dataset as a streaming dataset.
    Args:
        remote (str): Remote directory (S3 or local filesystem) where dataset is stored.
        local (str): Local filesystem directory where dataset is cached during operation.
        split (str): The dataset split to use. Currently, only ``None`` is supported.
        shuffle (bool): Whether to shuffle the samples in this dataset
        tokenizer_name_or_path (str): The name or path of the tokenizer to use. Default: ``'stabilityai/stable-diffusion-2-base'``.
        transform (Optional[Callable]): The transforms to apply to the image. Default: ``None``.
        predownload (Optional[int]): The number of samples to prefetch. Default: ``100_000``.
        download_retry (Optional[int]): The number of times to retry a download. Default: ``2``.
        download_timeout (Optional[float]): The timeout for a download. Default: ``120``.
        batch_size (Optional[int]): Hint batch_size that will be used on each device's DataLoader. Default: ``None``.
    """

    def __init__(
            self,
            remote: str,
            local: str,
            split: str = None,
            shuffle: bool = False,
            predownload: Optional[int] = 100_000,
            download_retry: Optional[int] = 2,
            download_timeout: Optional[float] = 120,
            batch_size: Optional[int] = None) -> None:
        super().__init__(remote=remote,
                         local=local,
                         split=split,
                         shuffle=shuffle,
                         predownload=predownload,
                         keep_zip=False,
                         download_retry=download_retry,
                         download_timeout=download_timeout,
                         validate_hash=None,
                         batch_size=batch_size)

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        return sample

def noop_collate(batch):
    return batch
dataset = StreamingLAIONDataset(remote="oci://mosaicml-internal-dataset-laion400m-unfiltered2", local="/tmp/mds-cache/unfiltered-laion/")
dataloader = DataLoader(dataset, batch_size=512, num_workers=64, collate_fn=noop_collate)

with streaming.MDSWriter(out="oci://mosaicml-internal-dataset-laion/filtered3", columns=columns, compression=None, hashes=[], size_limit=256*(2**20), max_workers=64) as writer:
    for i, sample_batch in tqdm(enumerate(dataloader)):
        for sample in sample_batch:
            if sample['status'] == 'success':
                writer.write(sample)



