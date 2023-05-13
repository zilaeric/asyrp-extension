#!/bin/bash

SH_FILE_NAME="script_train.sh"
GPU="0"

CONFIG="celeba.yml" # if you use other dataset, config/path_config.py should be matched
GUID="silly" # guid should be in utils/text_dic.py
CUDA_VISIBLE_DEVICES=$GPU

python main.py --run_train                        \
                        --config $CONFIG                                    \
                        --exp ./runs/$GUID                                  \
                        --edit_attr $GUID                                   \
                        --do_train 1                                        \
                        --do_test 1                                         \
                        --n_train_img 10                                   \
                        --n_test_img 10                                     \
                        --n_iter 5                                          \
                        --bs_train 1                                        \
                        --t_0 999                                           \
                        --n_inv_step 50                                     \
                        --n_train_step 50                                   \
                        --n_test_step 100                                   \
                        --get_h_num 1                                       \
                        --user_defined_t_edit 500                           \
                        --user_defined_t_addnoise 200                       \
                        --train_delta_block                                 \
                        --sh_file_name $SH_FILE_NAME                        \
                        --save_x0                                           \
                        --use_x0_tensor                                     \
                        --hs_coeff_delta_h 1.0                              \
                        --lr_training 0.5                                   \
                        --clip_loss_w 1.0                                   \
                        --l1_loss_w 3.0                                     \
                        --retrain 1                                         \
                        --custom_train_dataset_dir "/home/lcur1654/DL2-2023-group-15/data/celeba_hq/raw_images/train/images"                \
                        --custom_test_dataset_dir "/home/lcur1654/DL2-2023-group-15/data/celeba_hq/raw_images/test/images"                  \
                        --model_path "/home/lcur1654/models/asyrp/pretrained/celeba_hq.pt"
                        # --add_noise_from_xt                               \ # if you compute lpips, use it.
                        # --lpips_addnoise_th 1.2                           \ # if you compute lpips, use it.
                        # --lpips_edit_th 0.33                              \ # if you compute lpips, use it.
                        # --target_class_num $class_num                     \ # for imagenet

