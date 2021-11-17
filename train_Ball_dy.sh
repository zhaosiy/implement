python train_dy.py \
	--env Ball \
	--stage dy \
	--baseline 0 \
	--gauss_std 5e-2 \
	--lam_kp 10 \
	--en_model cnn \
	--dy_model gnn \
	--preload_kp 1 \
	--nf_hidden_kp 16 \
	--nf_hidden_dy 16 \
	--n_kp 5 \
	--inv_std 10 \
	--min_res 46 \
	--n_identify 100 \
	--n_his 10 \
	--n_roll 20 \
	--node_attr_dim 0 \
	--edge_attr_dim 1 \
	--edge_type_num 3 \
	--edge_st_idx 1 \
	--edge_share 1 \
	--batch_size 5 \
	--lr 1e-4 \
	--gen_data 0 \
	--num_workers 3 \
	--kp_epoch 2 \
	--kp_iter 10000 \
	--dy_epoch -1 \
	--dy_iter -1 \
	--action_dim 0 \
	--log_per_iter 50 \
	# --eval 1
