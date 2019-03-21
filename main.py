from Model import Model


if __name__ == '__main__':
    model_name = '20_paper_ep50_rst0'
    print(model_name)
    model = Model(dataset_name='cifar-10', model_name=model_name, n_epochs=50,
                  # block_width=256,
                  lr_segments=[(0.8, 1e-2), (0.1, 1e-3), (0.1, 1e-4)],
                  # lr_segments=[(0.25, 1e-5), (0.75, 1e-6)],
                  # lid_use_pre_relu=False, lda_use_pre_relu=True,
                  update_mode=1, update_param=4,
                  n_label_resets=1, cut_train_set=True,
                  log_mask=1 * (1 << 0) +
                           0 * (1 << 1) +
                           0 * (1 << 2) +
                           0 * (1 << 3)
                  )
    model.train(noise_ratio=0.2, noise_seed=0)

    # model_name = 'clean_none_lr_times_1e-3_4_block16'
    # # model_name = 'clean_paper_lr_1e-5_30_1e-6'
    # print(model_name)
    # # Model.compute_features(dataset_name='cifar-10', model_name=model_name, epoch=120, dataset_type='train',
    # #                        compute_pre_relu=True)
    # Model.compute_block_features(dataset_name='mnist', model_name=model_name, epoch=120, dataset_type='train',
    #                              n_blocks=4)
