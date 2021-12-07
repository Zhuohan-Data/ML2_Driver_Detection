#%%
def process_target(target_type,target):


    if target_type == 1:
        final_target = to_categorical(list(target))
    else:
        final_target = list(target)
    return final_target


def process_path(feature, target):
    # Processing Label

    label = target
    # Processing feature
    # load the raw data from the file as a string
    file_path = feature
    img = tf.io.read_file(file_path)

    img = tf.io.decode_jpeg(img, channels=CHANNELS)

    resized = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])


    return resized,label

def process_path_aug(feature, target):
    # Processing Label
    label = target
    # Processing feature
    # load the raw data from the file as a string
    file_path = feature
    img = tf.io.read_file(file_path)

    img = tf.io.decode_jpeg(img, channels=CHANNELS)


    #i = random.randrange(20)
    seed = (1, 5)
    img = tf.cast(img, tf.float32) / 255.0
    img_random_brightness = tf.image.stateless_random_brightness(img, max_delta=0.2, seed=seed)
    img_random_contrast = tf.image.stateless_random_contrast(img_random_brightness, lower=0.1, upper=0.2, seed=seed)
    img_random_crop = tf.image.stateless_random_crop(img_random_contrast, size=[IMAGE_SIZE-15, IMAGE_SIZE-15, 3], seed=seed)
    aug_img = tf.image.resize(img_random_crop, [IMAGE_SIZE, IMAGE_SIZE])

    return aug_img, label

def process_path_test(feature):
    # Processing Label

    # label = target
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
        #aug_targets = process_target(target_type, aug_targets)

        train_ds = tf.data.Dataset.from_tensor_slices((train_data, train_targets)).shuffle(buffer_size=1000, seed=random_state)
        val_ds = tf.data.Dataset.from_tensor_slices((val_data, val_targets)).shuffle(buffer_size=1000, seed=random_state)
        #aug_ds = tf.data.Dataset.from_tensor_slices((aug_data, aug_targets))

        final_train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)
        #aug_ds = aug_ds.map(process_path_aug, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)
        #final_train_ds = train_ds.concatenate(aug_ds)

        final_val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)

        return final_train_ds, final_val_ds

    if split == 'test':
        test_path = glob.glob(os.path.join(DATA_DIR, 'test', '*.jpg'))
        test_data = tf.data.Dataset.from_tensor_slices(test_path)
        final_test_ds = test_data.map(process_path_test, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)
        return final_test_ds




def model_definition():
    # Add the pretrained layers
    #pretrained_model = tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet')
    #pretrained_model = keras.applications.resnet_v2.ResNet152V2(include_top=False, weights='imagenet')
    #pretrained_model = keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet')
    #pretrained_model = keras.applications.densenet.DenseNet169(include_top=False, weights='imagenet')
    #pretrained_model = keras.applications.densenet.DenseNet201(include_top=False, weights='imagenet')
    #pretrained_model = tf.keras.applications.efficientnet.EfficientNetB2(include_top=False, weights='imagenet')
    #pretrained_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B2(include_top=False, weights='imagenet')
    #pretrained_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B3(include_top=False, weights='imagenet')
    #pretrained_model = tf.keras.applications.efficientnet_v2.EfficientNetV2M(include_top=False, weights='imagenet')
    #pretrained_model = tf.keras.applications.efficientnet_v2.EfficientNetV2L(include_top=False, weights='imagenet')
    pretrained_model = tf.keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet')
    # Add GlobalAveragePooling2D layer
    dropout = keras.layers.Dropout(rate=0.5)(pretrained_model.output)

    # Add GlobalAveragePooling2D layer
    average_pooling = keras.layers.GlobalAveragePooling2D()(dropout)

    # Add the output layer
    output = keras.layers.Dense(10, activation='softmax')(average_pooling)

    # Get the model
    model = keras.Model(inputs=pretrained_model.input, outputs=output)


    # learning rate scheduling
    #
    # lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate=0.01,
    #     decay_steps = s,
    #     decay_rate=0.1)

    #SGD = keras.optimizers.SGD(learning_rate=0.001)
    Adam = keras.optimizers.Adam(learning_rate=0.001)
    #Adadelta = keras.optimizers.Adadelta(learning_rate=0.001)
    model.compile(optimizer=Adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
#%%
 model_checkpoint_cb = keras.callbacks.ModelCheckpoint(filepath=os.path.join(os.getcwd(),'cache','Checkpoints'),
                                                          save_best_only=True,
                                                          save_weights_only=True)
    # EarlyStopping callback
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)
    # ReduceLROnPlateau callback
    reduce_lr_on_plateau_cb = keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=1)

    # Set the decay_steps
    # m = tf.data.experimental.cardinality(train_data).numpy()
    # s = int(20 * m / BATCH_SIZE)
    # print('Learning Scheduling Decay_Steps:', s)