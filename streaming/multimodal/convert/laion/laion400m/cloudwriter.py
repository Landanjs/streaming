# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Convert and upload LAION-400M parquet shards."""

import os
from argparse import ArgumentParser, Namespace
from time import sleep, time
from typing import Iterator, List, Optional, Union
from PIL import Image
from io import BytesIO
import warnings
import numpy as np
from pyarrow import parquet as pq
from tqdm import tqdm, trange
from multiprocessing import Pool
from functools import partial

from streaming import MDSWriter

# Change PIL image size warnings to be errors
warnings.filterwarnings("error", module='PIL', message='Image size')


def parse_args() -> Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace: Command-line arguments.
    """
    args = ArgumentParser()
    args.add_argument('--local',
                      type=str,
                      required=True,
                      help='Local directory containing downloaded shards in parquet format.')
    args.add_argument('--remote',
                      type=str,
                      default='',
                      help='Remote path to upload MDS-formatted shards to.')
    args.add_argument('--keep_parquet',
                      action='store_true',
                      help='Whether to keep the parquet shards after conversion (about 10TB).')
    args.add_argument('--hashes',
                      type=str,
                      default='sha1,xxh64',
                      help='Hashes for validating shards, if any.')
    args.add_argument('--poll_interval',
                      type=float,
                      default=30,
                      help='Interval between polling for newly downloaded shards to process.')
    args.add_argument('--bin_resolutions', action='store_true')
    return args.parse_args()


def filter_parquet_files(local: str) -> List:
    """List of parquet files to convert into MDS shards

    Args:
        local (str): Local directory containing shards.

    Returns:
        List[str]: Each parquet filename.
    """
    shards_to_process = []
    if not os.path.exists(local):
        print('Path does not exist!!')
        return shards_to_process
    for filename in os.listdir(local):
        # If _stats.json file is present, the parquet file has finished downloading
        if filename.endswith('_stats.json'):
            idx = filename.split('_')[0]

            # Check if parquet file has already been converted into an MDS shard
            done_filename = os.path.join(local, f'{idx}.done')
            done = os.path.exists(done_filename)
            if not done:
                shards_to_process.append(idx)


    return shards_to_process

def get_int(x: Union[float, int]) -> int:
    """Get an int field from pandas.

    Args:
        x (Union[float, int]): The pandas field.

    Returns:
        int: The normalized field.
    """
    if np.isnan(x):
        return 0
    else:
        return int(x)


def get_float(x: float) -> float:
    """Get a float field from pandas.

    Args:
        x (float): The pandas field.

    Returns:
        float: The normalized field.
    """
    return x


def get_bytes(x: Optional[bytes]) -> bytes:
    """Get a bytes field from pandas.

    Args:
        x (bytes, optional): The pandas field.

    Returns:
        float: The normalized field.
    """
    return x or b''


def get_str(x: Optional[str]) -> str:
    """Get a str field from pandas.

    Args:
        x (str, optional): The pandas field.

    Returns:
        str: The normalized field.
    """
    return x or ''

def convert_and_upload_shards(args: Namespace, writer) -> bool:
    """Process any newly downloaded shards.

    Args:
        args (Namespace): Command-line arguments.

    Returns:
        bool: Whether shard downloading is done.
    """
    hashes = args.hashes.split(',') if args.hashes else []
    # func = functools.partial(multi_proc, local=args.local, hashes=hashes, keep_parquet=args.keep_parquet, keep_mds=args.keep_mds)
    shards_to_process = filter_parquet_files(local=args.local)
    for shard in tqdm(shards_to_process):
        # Open parquet file
        parquet_filename = os.path.join(args.local, f'{shard}.parquet')
        table = pq.read_table(parquet_filename)
        n_rows = table.num_rows
        table = table.to_pandas()

        # Iterate through rows of parquet file
        for i in range(n_rows):
            x = table.iloc[i]

            # Only write samples that were successfully downloaded
            success = x['status'] == 'success'
            if success:
                try:
                    Image.open(BytesIO(x['jpg']))
                except Exception as e:
                    print(e)
                    # if unable to decode image, set success to false
                    success = False
            if success:
                sample = {
                    'punsafe': get_float(x['punsafe']),
                    'pwatermark': get_float(x['pwatermark']),
                    'similarity': get_float(x['similarity']),
                    'caption': get_str(x['caption']),
                    'url': get_str(x['url']),
                    'key': get_str(x['key']),
                    'status': get_str(x['status']),
                    'error_message': get_str(x['error_message']),
                    'width': get_int(x['width']),
                    'height': get_int(x['height']),
                    'original_width': get_int(x['original_width']),
                    'original_height': get_int(x['original_height']),
                    'exif': get_str(x['exif']),
                    'jpg': get_bytes(x['jpg']),
                    'hash': get_int(x['hash']),
                }
                if isinstance(writer, list):
                    writer[0].write(sample)
                    if sample['width'] >= 256 and sample['height'] >= 256:
                        writer[1].write(sample)
                else:
                    writer.write(sample)

        # Write .done file to indicate done with this parquet file
        done_filename = os.path.join(args.local, f'{shard}.done')
        with open(done_filename, 'w') as out:
            out.write('')

        # Delete parquet file
        if not args.keep_parquet:
            os.remove(parquet_filename)
        print(f'Shard {shard}: done')

    # Check if the done file was written and there are no more shards to process
    shards_to_process = filter_parquet_files(local=args.local)
    filename = os.path.join(args.local, 'done')
    done = os.path.exists(filename) and not shards_to_process
    return done, writer


def main(args: Namespace) -> None:
    """Convert and upload shards as they are created.

    Args:
        args (Namespace): Command-line arguments.
    """

    columns = {
        'punsafe': 'float64',
        'pwatermark': 'float64',
        'similarity': 'float64',
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
        'hash': 'int64',
    }

    if args.bin_resolutions:
        remote_all = os.path.join(args.remote, 'all')
        remote_256 = os.path.join(args.remote, '256')
        writer = []
        writer.append(MDSWriter(out=remote_all, columns=columns, compression=None, hash=[], size_limit=256*(2**20), max_workers=64))
        writer.append(MDSWriter(out=remote_256, columns=columns, compression=None, hash=[], size_limit=256*(2**20), max_workers=64))
    else:
        writer = MDSWriter(out=args.remote, columns=columns, compression=None, hash=[], size_limit=256*(2**20), max_workers=64)

    while True:
        last_poll = time()
        is_done, writer = convert_and_upload_shards(args, writer)
        if is_done:
            break
        now = time()
        elapsed = now - last_poll
        if elapsed < args.poll_interval:
            sleep(args.poll_interval - elapsed)
    if isinstance(writer, list):
        writer[0].finish()
        writer[1].finish()
    else:
        writer.finish()


if __name__ == '__main__':
    main(parse_args())
