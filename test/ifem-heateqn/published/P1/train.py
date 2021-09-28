from keras.models import Sequential
from keras.layers import Dense, LeakyReLU
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt

def make_model(): # create multiple identical models for comparison purposes
    model = Sequential()
    model.add(Dense(20,))
    for hidden_layers in range(4):
        model.add(Dense(80,))
        model.add(LeakyReLU(0.01))
    model.add(Dense(20,))

    opt = Adam(lr=1e-5)
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mae', 'mse'])
    return model


ham = make_model()
ddm = make_model()


# Read the training data
X  = np.load('train/X.npy')
Y  = np.load('train/Y.npy')
Z  = np.load('train/Z.npy')
Xv = np.load('validation/X.npy')
Yv = np.load('validation/Y.npy')
Zv = np.load('validation/Z.npy')


early_stopping_monitor = EarlyStopping(patience=20)
history_ham = ham.fit(X,Y,
                      batch_size=32,
                      epochs=100,
                      validation_data=(Xv,Yv),
                      callbacks=[early_stopping_monitor])


early_stopping_monitor = EarlyStopping(patience=20)
history_ddm = ddm.fit(X, Z,
                      batch_size=32,
                      epochs=100,
                      validation_data=(Xv,Zv),
                      callbacks=[early_stopping_monitor])

print(ham.summary())
ham.save('ham.h5')
ddm.save('ddm.h5')