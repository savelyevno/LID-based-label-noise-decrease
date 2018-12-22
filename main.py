from Model import Model


if __name__ == '__main__':
    # train_model('model25_batched_noLID', 'train25', False)
    # full_test('model')
    # test_image('testSample/img_3.jpg')

    model = Model(dataset_name='cifar-10', model_name='clean_lam_5e-4_lr_times_1e-1_aug',
                  update_mode=0, update_param=0, update_submode=0, update_subsubmode=0,
                  log_mask=1 * (1 << 0) +
                           0 * (1 << 1) +
                           0 * (1 << 2) +
                           0 * (1 << 3) +
                           0 * (1 << 4)
                  )
    model.train('train')
