from Model import Model


if __name__ == '__main__':
    # train_model('model25_batched_noLID', 'train25', False)
    # full_test('model')
    # test_image('testSample/img_3.jpg')

    model = Model('40_cosine_pre_lid_5', update_mode=2, update_param=5, update_submode=0, update_subsubmode=0,
                  log_mask=1 * (1 << 0) +
                           1 * (1 << 1) +
                           0 * (1 << 2) +
                           0 * (1 << 3) +
                           0 * (1 << 4) +
                           0 * (1 << 5) +
                           0 * (1 << 6)
                  )
    model.train('train40')
