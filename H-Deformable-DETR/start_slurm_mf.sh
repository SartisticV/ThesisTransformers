#!/usr/bin/env bash

GPUS_PER_NODE=4 ./tools/run_dist_slurm.sh gpua16 h_deformable_detr 4 configs/two_stage/deformable-detr-hybrid-branch/36eps/swin/swin_tiny_hybrid_branch_lambda1_group6_t1500_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage.sh \
--resume ./weights/swin_tiny_hybrid_branch_lambda1_group6_t1500_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_36eps.pth \
--pretrained_backbone_path ./weights/swin_tiny_hybrid_branch_lambda1_group6_t1500_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_36eps.pth \
--backbone video_swin_tiny \
--pretrained2d \
--fusion enc \
--num_ref_frames 5 \
--stride 1
