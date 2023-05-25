#!/bin/bash

sh_file_name="script_inference.sh"
gpu="0"
config="celeba.yml"
guid="young"
dt_lambda=1.0   # hyperparameter for dt_lambda. This is the method that will appear in the next paper.
CUDA_VISIBLE_DEVICES=$gpu

python main.py  --run_test                                                     \
                --config $config                                               \
                --exp ../../runs/${guid}                                       \
                --edit_attr $guid                                              \
                --do_train 1                                                   \
                --do_test 1                                                    \
                --n_train_img 0                                                \
                --n_test_img 100                                               \
                --n_iter 1                                                     \
                --bs_train 1                                                   \
                --n_inv_step 40                                                \
                --n_train_step 40                                              \
                --n_test_step 40                                               \
                --get_h_num 1                                                  \
                --train_delta_block                                            \
                --sh_file_name $sh_file_name                                   \
                --save_x0                                                      \
                --use_x0_tensor                                                \
                --hs_coeff_delta_h 1.0                                         \
                --dt_lambda $dt_lambda                                         \
                --manual_checkpoint_name "young_LC_CelebA_HQ_t999_ninv40_ngen40_0.pth"    \
                --add_noise_from_xt                                            \
                --user_defined_t_edit 513                                      \
                --user_defined_t_addnoise 167                                  \
                --save_process_origin                                          \
                --save_x_origin 

                # if you did not compute lpips, use it.
                # --user_defined_t_edit 500                                    \
                # --user_defined_t_addnoise 200                                \

