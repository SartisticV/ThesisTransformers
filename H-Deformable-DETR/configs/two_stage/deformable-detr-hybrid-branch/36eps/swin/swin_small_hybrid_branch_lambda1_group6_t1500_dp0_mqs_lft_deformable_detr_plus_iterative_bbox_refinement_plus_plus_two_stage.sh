#!/usr/bin/env bash

set -x

EXP_DIR=exps/two_stage/deformable-detr-hybrid-branch/36eps/swin/swin_small_hybrid_branch_lambda1_group6_t1500_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage

python -u main.py \
    --output_dir ${EXP_DIR} \
    --with_box_refine \
    --two_stage \
    --dim_feedforward 2048 \
    --epochs 36 \
    --lr_drop 30 \
    --num_queries_one2one 100 \
    --num_queries_one2many 500 \
    --k_one2many 6 \
    --lambda_one2many 1.0 \
    --dropout 0.0 \
    --mixed_selection \
    --look_forward_twice \
    --backbone swin_small \
    --pretrained_backbone_path /mnt/pretrained_backbone/swin_small_patch4_window7_224.pth \
    ${PY_ARGS}
