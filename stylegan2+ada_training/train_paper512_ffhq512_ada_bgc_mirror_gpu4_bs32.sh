#!/bin/bash

python3 train.py \
	--outdir=models/dfuc2022/clean_baseline_train_cc512___paper512_ffhq512_ada_bgc_mirror_gpu4_bs32 \
	--data=datasets/dfuc2022/clean_baseline_train_cc512.zip \
	--cfg=paper512 \
	--resume=ffhq512 \
	--augpipe=bgc \
	--mirror=true \
	--gpus=4 \
	--batch=32 \
	--snap=10 \
	--kimg=3000 \
	--metrics=fid50k_full
