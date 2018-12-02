from Model import Model


if __name__ == '__main__':
    # train_model('model25_batched_noLID', 'train25', False)
    # full_test('model')
    # test_image('testSample/img_3.jpg')

    model = Model('model25_relu', update_mode=0, log_mask=1)
    model.train('train25')
