from Model import Model


if __name__ == '__main__':
    # train_model('model25_batched_noLID', 'train25', False)
    # full_test('model')
    # test_image('testSample/img_3.jpg')

    model_name = '20_lda_3_relu_lam_1e-3_lr_times_1e-1_not_aug'
    print(model_name)
    model = Model(dataset_name='cifar-10', model_name=model_name,
                  update_mode=3, update_param=5, update_submode=1, update_subsubmode=0, reg_coef=1e-3,
                  log_mask=1 * (1 << 0) +
                           0 * (1 << 1) +
                           0 * (1 << 2) +
                           0 * (1 << 3) +
                           0 * (1 << 4)
                  )
    model.train('train20')
