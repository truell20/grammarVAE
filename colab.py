import h5py
import tensorflow as tf
with tf.device('/gpu:0'):

    qm9_data_path = 'qm9_two_tower_dataset.h5'
    h5f = h5py.File(qm9_data_path, 'r')
    functional_data = h5f['functional_data'][:]
    grammar_data = h5f['grammar_data'][:]
    h5f.close()

    import zinc_grammar as G

    rules = G.gram.split('\n')
    MAX_LEN = 150
    DIM = len(rules)
    LATENT = 64
    EPOCHS = 100
    BATCH = 1024

    from models.model_zinc import MoleculeVAE
    import os

    # 2. get any arguments and define save file, then create the VAE model
    print('L='  + str(LATENT) + ' E=' + str(EPOCHS))
    save_path = 'save_dir'
    model_save = save_path+'qm9_vae_grammar_L' + str(LATENT) + '_E' + str(EPOCHS) + '_val.hdf5'
    print(model_save)
    model = MoleculeVAE()

    # 3. if this results file exists already load it
    load_model=''
    if os.path.isfile(load_model):
        print('Loading...')
        model.load(rules, load_model, latent_rep_size = LATENT, max_length=MAX_LEN)
        print('Done loading!')
    else:
        print('Making new model...')
        model.create(rules, max_length=MAX_LEN, latent_rep_size = LATENT)
        print('New model created!')

    from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

    # 4. only save best model found on a 10% validation set
    checkpointer = ModelCheckpoint(filepath = model_save,
                                    verbose = 2,
                                    save_best_only = True)

    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                                    factor = 0.2,
                                    patience = 3,
                                    min_lr = 0.0001)

    early_stop = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=10,
        mode='min'
    )

    # model.autoencoder.summary()

    # 5. fit the vae
    history=model.autoencoder.fit(
        [grammar_data, functional_data],
        [grammar_data, functional_data],
        batch_size=BATCH,
        shuffle=True,
        epochs=EPOCHS,
        verbose=1,
        callbacks = [checkpointer, reduce_lr, early_stop],
        validation_split = 0.1)
