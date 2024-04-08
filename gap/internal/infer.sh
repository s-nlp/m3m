#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python cli_gap.py \
        --do_predict \
        --output_dir out \
        --predict_file data/mintaka_test \
        --tokenizer_path facebook/bart-base \
        --dataset webnlg \
	--entity_entity \
        --entity_relation \
	--type_encoding \
	--max_node_length 60 \
        --predict_batch_size 16 \
        --max_input_length 256 \
        --max_output_length 512 \
        --append_another_bos \
        --num_beams 5 \
	      --length_penalty 5 \
