#%%
os.system("sudo pip install tf-nightly-gpu")
# CSV_DIR = os.getcwd()+os.path.sep+'driver_imgs_list.csv'
DATA_DIR = os.getcwd()+os.path.sep+'data' + os.path.sep +'imgs' + os.path.sep
CSV_DIR = os.getcwd()+os.path.sep+'data'+os.path.sep+'driver_imgs_list.csv'
sep = os.path.sep

nfolds=5
n_epoch = 10
BATCH_SIZE = 16

#%%
def process_path_test(feature):
    # Processing feature
    # load the raw data from the file as a string
    file_path = feature
    img = tf.io.read_file(file_path)

    img = tf.io.decode_jpeg(img, channels=CHANNELS)

    resized = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
    return resized

def read_data(target_type,split='train',train_index=0,valid_index=0,method='split'):

    if split=='train':

        if method=='split':
            train_data,val_data,train_targets,val_targets =split_validation_set(ds_inputs,ds_targets,0.2)
        else:
            train_data, val_data = ds_inputs[train_index], ds_inputs[valid_index]
            train_targets, val_targets = ds_targets[train_index], ds_targets[valid_index]

        #nope_data, aug_data, nope_targets, aug_targets = split_validation_set(train_data, train_targets, 0.1)

        train_targets = process_target(target_type, train_targets)
        val_targets = process_target(target_type, val_targets)

        train_ds = tf.data.Dataset.from_tensor_slices((train_data, train_targets)).shuffle(buffer_size=1000, seed=random_state)
        val_ds = tf.data.Dataset.from_tensor_slices((val_data, val_targets)).shuffle(buffer_size=1000, seed=random_state)

        final_train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)

        final_val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)

        return final_train_ds, final_val_ds

    if split == 'test':
        test_path = glob.glob(os.path.join(DATA_DIR, 'test', '*.jpg'))
        test_data = tf.data.Dataset.from_tensor_slices(test_path)
        final_test_ds = test_data.map(process_path_test, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)
        return final_test_ds
#%%
def run_cross_validation(nfolds=10, nb_epoch=10, split=0.2, modelStr='', method='split' ):


    # ModelCheckpoint callback
    if not os.path.isdir('Checkpoints'):
        os.mkdir('Checkpoints')
    model_checkpoint_cb = keras.callbacks.ModelCheckpoint(filepath=os.path.join(os.getcwd(),'cache','Checkpoints'),
                                                          save_best_only=True,
                                                          save_weights_only=True)
    # EarlyStopping callback
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)
    # ReduceLROnPlateau callback
    reduce_lr_on_plateau_cb = keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=1)

    # Kfold
    num_fold = 0
    kf = KFold(n_splits=nfolds,
               shuffle=True, random_state=random_state)
    val_acc_best = 0

    # Set the decay_steps
    # m = tf.data.experimental.cardinality(train_data).numpy()
    # s = int(20 * m / BATCH_SIZE)
    # print('Learning Scheduling Decay_Steps:', s)

    if method=='split':
        model = model_definition()
        train_data, valid_data = read_data(target_type=1, split='train',
                                           method=method)
        history = model.fit(train_data,
                            epochs=nb_epoch,
                            validation_data=valid_data,
                            verbose=1, shuffle=True,)
        save_model(model, num_fold, modelStr)
    else:
        for train_index, valid_index in kf.split(ds_inputs):
            num_fold += 1

            train_data, valid_data = read_data(target_type=1, split='train',
                                               train_index=train_index,valid_index=valid_index,
                                               method='kfold')





            model = model_definition()
            print('Start KFold number {} from {}'.format(num_fold, nfolds))
            print(len(train_data))
            history = model.fit(train_data,
                      # batch_size=batch_size,
                      epochs=nb_epoch,
                      validation_data = valid_data,
                      verbose=1, shuffle=True,
                      callbacks=[model_checkpoint_cb,
                                       early_stopping_cb,
                                 reduce_lr_on_plateau_cb])

            # Kfold best
            # if val_acc_best < max(history.history['val_accuracy']):
            #     val_acc_best = max(history.history['val_accuracy'])
            #     save_model(model, 'best', modelStr)
            #     print('best model saved')

            # Kfold Ensemble
            save_model(model, num_fold, modelStr)
#%%
def test_model_and_submit(start=1, end=1, modelStr=''):
    img_rows, img_cols = IMAGE_SIZE,IMAGE_SIZE


    print('Start testing............')
    test_data = read_data(target_type=1, split='test')

    test_path = os.path.join(DATA_DIR, 'test', '*.jpg')
    images = glob.glob(test_path)

    test_ids = []
    for img_path in images:
        img_id = os.path.basename(img_path)
        test_ids.append(img_id)


    yfull_test = []

    for index in range(start, end + 1):
        # Store test predictions
        model = read_model(index, modelStr)
        # test_prediction = model.predict(test_data, batch_size=32, verbose=1)
        test_prediction = model.predict(test_data,verbose=1)

        yfull_test.append(test_prediction)

    info_string = 'loss_' + modelStr \
                  + '_r_' + str(img_rows) \
                  + '_c_' + str(img_cols) \
                  + '_folds_' + str(end - start + 1)
    # model = read_model('best',modelStr)
    # test_res = model.predict(test_data)
    test_res = merge_several_folds_mean(yfull_test, end - start + 1)
    create_submission(test_res, test_ids, info_string)
    print('finished')

def test_ensemble_and_submit(start=1, end=1, modelStr='',modelnames=[1,2],method='split'):
    img_rows, img_cols = IMAGE_SIZE,IMAGE_SIZE
    # batch_size = 64
    # random_state = 51

    print('Start testing............')
    test_data = read_data(target_type=1, split='test')

    test_path = os.path.join(DATA_DIR, 'test', '*.jpg')
    images = glob.glob(test_path)

    test_ids = []
    for img_path in images:
        img_id = os.path.basename(img_path)
        test_ids.append(img_id)


    yfull_test = []
    if method=='split':
        model1 = read_model(1,modelnames[0])
        test_prediction = model1.predict(test_data, verbose=1)

        yfull_test.append(test_prediction)

        model2 = read_model(1,modelnames[1])
        test_prediction = model2.predict(test_data, verbose=1)
        yfull_test.append(test_prediction)

        test_res = merge_several_folds_mean(yfull_test, 2)
    else:
        for i in modelnames:
            for index in range(start, end + 1):
                # Store test predictions
                model = read_model(index, i)
                test_prediction = model.predict(test_data,verbose=1)

                yfull_test.append(test_prediction)

        test_res = merge_several_folds_mean(yfull_test, (end - start + 1)*len(modelnames))
    info_string = 'loss_' + modelStr \
                  + '_r_' + str(img_rows) \
                  + '_c_' + str(img_cols) \
                  + '_folds_' + str(end - start + 1)
    # model = read_model('best',modelStr)
    # test_res = model.predict(test_data)
    create_submission(test_res, test_ids, info_string)
    print('finished')