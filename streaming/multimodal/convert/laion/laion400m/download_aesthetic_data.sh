#!/bin/sh

if hash wandb 2> /dev/null; then
    wandb login
    ENABLE_WANDB=True
else
    ENABLE_WANDB=False
fi

img2dataset \
    --url_list /tmp/laion2b-raw \
    --input_format parquet \
    --url_col URL \
    --caption_col TEXT \
    --output_format parquet \
    --output_folder /tmp/laion2b-processed \
    --processes_count 48 \
    --thread_count 96 \
    --resize_mode no \
    --save_additional_columns '["punsafe","pwatermark","similarity","hash"]' \
    --enable_wandb True

touch /tmp/laion2b-processed/done
