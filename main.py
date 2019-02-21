from Model import Model


if __name__ == '__main__':
    # train_model('model25_batched_noLID', 'train25', False)
    # full_test('model')
    # test_image('testSample/img_3.jpg')

    model_name = '20_lda_lr_times_1e-1_fc32'
    print(model_name)
    model = Model(dataset_name='cifar-10', model_name=model_name,
                  update_mode=3, update_param=5, update_submode=0, update_subsubmode=0, reg_coef=0,
                  log_mask=1 * (1 << 0) +
                           0 * (1 << 1) +
                           0 * (1 << 2) +
                           0 * (1 << 3) +
                           0 * (1 << 4)
                  )
    model.train('train20')

    # model_name = 'clean_paper_lam_1e-3_lr_times_1e-1_aug'
    # # model_name = 'clean_paper_lr_1e-5_30_1e-6'
    # print(model_name)
    # Model.compute_features(dataset_name='cifar-10', model_name=model_name, epoch=120, dataset_type='train',
    #                        compute_pre_relu=True)
