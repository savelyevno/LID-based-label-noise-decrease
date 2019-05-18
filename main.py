from Model import Model


if __name__ == '__main__':
    dataset_name = 'cifar-100'
    model_name = '60_paperSepLL_ep100_init95_sep_LL_2_75_64_startAfterInitEpoch'
    print(dataset_name, model_name)
    model = Model(dataset_name=dataset_name, model_name=model_name,
                  # reg_coef=1e-6, lr_segments=[(80, 1e-2), (5, 1e-3), (40, 1e-4), (40, 1e-5)],      # cifar-100 165
                  reg_coef=1e-6, lr_segments=[(80, 1e-2), (5, 1e-3), (15, 1e-4)],      # cifar-100 60%
                  # reg_coef=1e-4, lr_segments=[(30, 1e-2), (10, 1e-3), (10, 1e-4)],       # cifar-10 50
                  # reg_coef=0, lr_segments=[(20, 1e-4), (20, 1e-5), (10, 1e-6)],          # mnist paper
                  # reg_coef=1e-4, lr_segments=[(40, 1e-2), (40, 1e-3), (40, 1e-4)],      # cifar-10 paper
                  # reg_coef=1e-4, lr_segments=[(40, 1e-2), (1, 1e-3), (39, 1e-4)],      # cifar-10 improved
                  # reg_coef=1e-6, lr_segments=[(80, 1e-2), (40, 1e-3), (40, 1e-4), (40, 1e-5)],  # cifar-100 paper
                  # lid_use_pre_relu=False, lda_use_pre_relu=True,
                  update_mode=4,
                  # calc_lid_min_before_init_epoch=True,
                  start_after_init_epoch=True,
                  init_epochs=95,
                  n_label_resets=0, cut_train_set=True, mod_labels_after_last_reset=True, use_loss_weights=False,
                  train_separate_ll=True, separate_ll_class_count=2, separate_ll_count=75, separate_ll_fc_width=64,
                  log_mask=1 * (1 << 0) +
                           0 * (1 << 1) +
                           0 * (1 << 2) +
                           0 * (1 << 3) +
                           0 * (1 << 4) +
                           0 * (1 << 5) +
                           0 * (1 << 6))
    model.train(noise_ratio=0.6, noise_seed=0)

    # model_name = 'clean_none_lr_times_1e-3_4_block16'
    # # model_name = 'clean_paper_lr_1e-5_30_1e-6'
    # print(model_name)
    # # Model.compute_features(dataset_name='cifar-10', model_name=model_name, epoch=120, dataset_type='train',
    # #                        compute_pre_relu=True)
    # Model.compute_block_features(dataset_name='mnist', model_name=model_name, epoch=120, dataset_type='train',
    #                              n_blocks=4)
