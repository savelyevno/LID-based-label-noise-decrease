from Model import Model


if __name__ == '__main__':
    # train_model('model25_batched_noLID', 'train25', False)
    # full_test('model')
    # test_image('testSample/img_3.jpg')

    model = Model('model25_2_clipped_2stdev_weighted_mean', lid_update_mode=2, lid_log_mask=3)
    model.train('train25')
