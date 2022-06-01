import tensorflow as tf
import LRSchedular
from sklearn.model_selection import train_test_split
import Dataset as D
import DataLoader as DL
import Model2 as M
import os
import numpy as np
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")


if __name__ == '__main__':

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    X_train, X_test, y_train, y_test = D.get_splitted_datas()
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.1)
    val_generator = DL.get_generator(X_val, y_val, 32)
    test_generator = DL.get_generator(X_test, y_test, 1)
    X_test = []
    y_test = []
    for i in test_generator:
        X_test.append(i[0][0, :, :, :])
        y_test.append(i[1][0, :, :, :])
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    train_generator = DL.get_generator(X_train, y_train, 32)

    clr = LRSchedular.CyclicLR(base_lr=1e-5, max_lr=1e-3,
                   step_size=4000., mode='triangular2')

    early_stop = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', min_delta=0, patience=5, verbose=0,
                    mode='auto', baseline=None, restore_best_weights=True
    )






    # model = M.unet()
    # model.fit(train_generator, validation_data=val_generator, epochs=500, verbose=1, shuffle=False, callbacks=[early_stop])
    # model.save('record9/')
    model = tf.keras.models.load_model('record9/', compile =False)
    # model.load_weights('record5/')
    # model.fit(train_generator, validation_data=val_generator, epochs=2, verbose=1, shuffle=False, callbacks=[clr])
    # model.save('record6/')
    y_pred = model.predict(X_test)

    import matplotlib.pyplot as plt
    row = 4
    col = 3
    fig1 = plt.figure(1, figsize=(200, 200))


    for i in range(1, col * row + 1):
        fig1.add_subplot(row, col, i)
        fig1.set_size_inches(18.5, 10.5, forward=True)
        xx = y_pred[i]
        xx[xx >= 0.5] = 1
        xx[xx < 0.5] = 0
        xx = xx.reshape((256, 256, 1))
        res = np.hstack((xx, y_test[i]))
        plt.imshow(res, cmap='gray')

    fig2 = plt.figure(2, figsize=(200, 200))
    for i in range(1, col * row + 1):
        fig2.add_subplot(row, col, i)
        fig2.set_size_inches(18.5, 10.5, forward=True)
        xx = y_pred[12 + i]
        xx[xx >= 0.5] = 1
        xx[xx < 0.5] = 0
        xx = xx.reshape((256, 256, 1))
        res = np.hstack((xx, y_test[12 + i]))
        plt.imshow(res, cmap='gray')
    plt.show()


