from Model import Model


if __name__ == '__main__':
    dataset_name = 'cifar-100'
    model_name = '20_paper_ep100_init30_fc64'
    print(dataset_name, model_name)
    model = Model(dataset_name=dataset_name, model_name=model_name, n_epochs=100, reg_coef=1e-6,
                  # block_width=256,
                  # lr_segments=[(0.4, 1e-2), (0.2, 1e-3), (0.2, 1e-4), (0.2, 1e-5)],
                  lr_segments=[(0.57, 1e-2), (0.14, 1e-3), (0.28, 1e-4)],
                  # lr_segments=[(0.8, 1e-2), (0.1, 1e-3), (0.1, 1e-4)],
                  # lr_segments=[(0.25, 1e-5), (0.75, 1e-6)],
                  # lid_use_pre_relu=False, lda_use_pre_relu=True,
                  update_mode=1,
                  init_epochs=30,
                  n_label_resets=0, cut_train_set=False, mod_labels_after_last_reset=True, use_loss_weights=False,
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
