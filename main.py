from Model import Model


if __name__ == '__main__':
    dataset_name = 'cifar-100'
    model_name = '20_none_ep50_cifar10Model'
    print(dataset_name, model_name)
    model = Model(dataset_name=dataset_name, model_name=model_name, n_epochs=50, reg_coef=1e-4,
                  # block_width=256,
                  # lr_segments=[(0.4, 1e-2), (0.2, 1e-3), (0.2, 1e-4), (0.2, 1e-5)],
                  # lr_segments=[(55, 1e-2), (15, 1e-3), (30, 1e-4)],                 # cifar-100 100
                  lr_segments=[(30, 1e-2), (10, 1e-3), (10, 1e-4)],                 # cifar-10 50
                  # lr_segments=[(20, 1e-4), (20, 1e-5), (10, 1e-6)],                 # mnist paper
                  # lr_segments=[(40, 1e-2), (40, 1e-3), (40, 1e-4)],                 # cifar-10 paper
                  # lr_segments=[(80, 1e-2), (40, 1e-3), (40, 1e-4), (40, 1e-5)],     # cifar-100 paper
                  # lr_segments=[(0.25, 1e-5), (0.75, 1e-6)],
                  # lid_use_pre_relu=False, lda_use_pre_relu=True,
                  update_mode=0,
                  init_epochs=40,
                  n_label_resets=0, cut_train_set=False, mod_labels_after_last_reset=True, use_loss_weights=False,
                  calc_lid_min_before_init_epoch=False,
                  log_mask=1 * (1 << 0) +
                           0 * (1 << 1) +
                           0 * (1 << 2) +
                           0 * (1 << 3))
    model.train(noise_ratio=0.2, noise_seed=0)

    # model_name = 'clean_none_lr_times_1e-3_4_block16'
    # # model_name = 'clean_paper_lr_1e-5_30_1e-6'
    # print(model_name)
    # # Model.compute_features(dataset_name='cifar-10', model_name=model_name, epoch=120, dataset_type='train',
    # #                        compute_pre_relu=True)
    # Model.compute_block_features(dataset_name='mnist', model_name=model_name, epoch=120, dataset_type='train',
    #                              n_blocks=4)
