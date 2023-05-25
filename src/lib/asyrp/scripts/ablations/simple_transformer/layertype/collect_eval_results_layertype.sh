#!/bin/bash

sh_file_name="script_train.sh" # nvm this thing
gpu="0"

config="celeba.yml" # if you use other dataset, config/path_config.py should be matched
guid="neanderthal" # guid should be in utils/text_dic.py
CUDA_VISIBLE_DEVICES=$gpu

layertypes=("c_transformer_simple" "cp_transformer_simple" "pc_transformer_simple" "p_transformer_simple")

for layer in "${layertypes[@]}"
do
    for epoch in {0..3}
    do
        python main.py  --run_test                     \
        --config $config                \
        --exp ../../eval_runs/layer_abl_transf_${layer}_h1_l1_d2048_${guid}          \
        --manual_checkpoint_name /home/parting/master_AI/DL2/ablations/layertype/${layer}/transf_${layer}_h1_l1_d2048_neanderthal_LC_CelebA_HQ_t999_ninv40_ngen40_${epoch}.pth \
        --edit_attr $guid               \
        --do_train 0                    \
        --do_test 1                     \
        --bs_train 1                    \
        --bs_test 1                     \
        --n_train_img 1000              \
        --accumulation_steps 1          \
        --n_test_img 100                \
        --n_inv_step 40                 \
        --n_train_step 40               \
        --n_test_step 40                \
        --get_h_num 1                   \
        --train_delta_block             \
        --sh_file_name $sh_file_name    \
        --n_iter 20                     \
        --save_x0                       \
        --use_x0_tensor                 \
        --save_x_origin                 \
        --clip_loss_w 0.8               \
        --l1_loss_w 3.0                 \
        --db_layer_type "${layer}"      \
        --db_nheads 1                   \
        --db_num_layers 1               \
        --db_dim_feedforward 2048       \
        # --load_random_noise             \
        # --user_defined_t_edit 513       \
        # --user_defined_t_addnoise 167   \
        mkdir -p ../../eval_runs/layertype_ablation/${layer}/${epoch}
        cp -r ../../eval_runs/layer_abl_transf_${layer}_h1_l1_d2048_${guid}_LC_CelebA_HQ_t999_ninv40_ngen40/test_images/40 ../../eval_runs/layertype_ablation/${layer}/${epoch}
    done
done

