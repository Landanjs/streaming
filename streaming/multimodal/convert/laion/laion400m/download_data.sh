#!/bin/sh

if hash wandb 2> /dev/null; then
    wandb login
    ENABLE_WANDB=True
else
    ENABLE_WANDB=False
fi

img2dataset \
    --url_list /workdisk/landan/laion/laion400m-meta \
    --input_format parquet \
    --url_col URL \
    --caption_col TEXT \
    --output_format parquet \
    --output_folder /tmp/landan-laion400m-data \
    --processes_count 64 \
    --thread_count 256 \
    --resize_mode no \
    --image_size 256 \
    --min_image_size 256 \
    --save_additional_columns '["NSFW","similarity","LICENSE"]' \
    --enable_wandb True

touch /tmp/landan-laion400m-data/done
