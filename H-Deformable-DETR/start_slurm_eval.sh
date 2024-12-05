#!/usr/bin/env bash

GPUS_PER_NODE=1 ./tools/run_dist_slurm.sh gpua16 h_deformable_detr 1 configs/two_stage/deformable-detr-hybrid-branch/36eps/swin/swin_tiny_hybrid_branch_lambda1_group6_t1500_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage.sh \
--resume /scratch/6409458/data/weights/checkpoint0003.pth --pretrained_backbone_path /scratch/6409458/data/weights/checkpoint0003.pth --fusion sf --eval --backbone swin_tiny