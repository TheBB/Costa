from keras.models import Sequential
from keras.layers import Dense, LeakyReLU
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import sys
import numpy as np
import matplotlib.pyplot as plt

def make_model(): # create multiple identical models for comparison purposes
    model = Sequential()
    model.add(Dense(20,))
    for hidden_layers in range(4):
        model.add(Dense(80,))
        model.add(LeakyReLU(0.01))
    model.add(Dense(20,))

    opt = Adam(learning_rate=1e-5)
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mae', 'mse'])
    return model


example_name = sys.argv[1]

ham = make_model()
ddm = make_model()


# Read the training data
X    = np.load(f'{example_name}/train/X.npy')
Y    = np.load(f'{example_name}/train/Y.npy')
Xex  = np.load(f'{example_name}/train/Xex.npy')
Yex  = np.load(f'{example_name}/train/Yex.npy')
Xv   = np.load(f'{example_name}/validation/X.npy')
Yv   = np.load(f'{example_name}/validation/Y.npy')
Xexv = np.load(f'{example_name}/validation/Xex.npy')
Yexv = np.load(f'{example_name}/validation/Yex.npy')


early_stopping_monitor = EarlyStopping(patience=20)
history_ham = ham.fit(X,Y,
                      batch_size=32,
                      epochs=100,
                      validation_data=(Xv,Yv),
                      callbacks=[early_stopping_monitor])


early_stopping_monitor = EarlyStopping(patience=20)
history_ddm = ddm.fit(Xex, Yex,
                      batch_size=32,
                      epochs=100,
                      validation_data=(Xexv,Yexv),
                      callbacks=[early_stopping_monitor])

print(ham.summary())
ham.save(f'{example_name}/ham.h5')
ddm.save(f'{example_name}/ddm.h5')
