DATA_PATH="YOUR PATH"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch \
--master_port 6089 \
--nproc_per_node=8 main.py \
--do_train 1 \
--workers 8 \
--n_display 50 \
--epochs 5 \
--lr 1e-4 \
--coef_lr 1e-3 \
--batch_size 128 \
--batch_size_val 128 \
--anno_path ${DATA_PATH}/ \
--video_path ${DATA_PATH}/MSRVTT_Videos \
--datatype msrvtt \
--max_words 32 \
--max_frames 12 \
--video_framerate 1 \
--base_encoder ViT-B/32 \
--agg_module seqTransf \
--interaction wti \
--wti_arch 2 \
--output_dir ckpts/ckpt_KDProR_MSR-VTT_Baseline \
--knowledge_scale 0.2 \
--lambda_1 0.2 \
--lambda_2 0.3 \
--model_type DRL # DRL-WTI as baseline
