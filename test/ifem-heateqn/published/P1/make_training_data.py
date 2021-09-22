import IFEM_CoSTA
import numpy as np
import matplotlib.pyplot as plt
import splipy as sp
import sys
from os import mkdir, path, listdir

# split alpha values into disjoint test, train and validation sets
train      = [    .1, .2, .3, .4, .5, .6,         .9, 1.0,      1.2, 1.3, 1.4,      1.6, 1.7, 1.8, 1.9, 2.0       ]
validation = [                                .8,          1.1                                                    ]
test       = [-.5,                        .7,                                  1.5,                          2.5  ]

# create file structure test/train with all input files in it
if not path.isdir('train'):
    mkdir('train')
if not path.isdir('test'):
    mkdir('test')
if not path.isdir('validation'):
    mkdir('validation')

########################################################################

#             CREATE THE TRAINING DATA                                 #

########################################################################
# initialize the simulation
ndof = 20
nt   = 5001
X = np.zeros((ndof  , len(train)*nt))
Y = np.zeros((ndof  , len(train)*nt))
i = 0
for alpha in train:
    # set up physics based model
    full = IFEM_CoSTA.HeatEquation('P1.xinp')
    mu = {'dt':0.001, 't':0.0, 'ALPHA':alpha}
    uprev = full.initial_condition(mu)
    zeros = [0.0]*full.ndof

    # run time iterations
    for n in range(5001):
        upred = full.predict(mu, uprev)

        sigma = full.residual(mu, uprev, upred)
        X[:,i] = upred
        Y[:,i] = sigma
        i += 1

        # this is training, so we have the correct residual
        ucorr = full.correct(mu, uprev, zeros)
        uprev = ucorr
        mu['t'] += mu['dt']

# save results to file
np.save('train/X.npy', X.T)
np.save('train/Y.npy', Y.T)



########################################################################

#             CREATE THE VALUDATION DATA                               #

########################################################################
# This is *exactly* the same as above with the word 'train', changed
# for 'validation'
ndof = 20
nt   = 5001
X = np.zeros((ndof  , len(validation)*nt))
Y = np.zeros((ndof  , len(validation)*nt))
i = 0
for alpha in validation:
    # set up physics based model
    full = IFEM_CoSTA.HeatEquation('P1.xinp')
    mu = {'dt':0.001, 't':0.0, 'ALPHA':alpha}
    uprev = full.initial_condition(mu)
    zeros = [0.0]*full.ndof

    # run time iterations
    for n in range(5001):
        upred = full.predict(mu, uprev)

        sigma = full.residual(mu, uprev, upred)
        X[:,i] = upred
        Y[:,i] = sigma
        i += 1

        # this is training, so we have the correct residual
        ucorr = full.correct(mu, uprev, zeros)
        uprev = ucorr
        mu['t'] += mu['dt']

# save results to file
np.save('validation/X.npy', X.T)
np.save('validation/Y.npy', Y.T)


########################################################################

#             CREATE THE TEST DATA                                     #

########################################################################
# initialize the simulation
# This actually does something else as it splits the data into chunks
ndof = 20
nt   = 5001
X = np.zeros((ndof  , nt))
Y = np.zeros((ndof  , nt))
for alpha in test:
    # set up physics based model
    full = IFEM_CoSTA.HeatEquation('P1.xinp')
    mu = {'dt':0.001, 't':0.0, 'ALPHA':alpha}
    uprev = full.initial_condition(mu)
    zeros = [0.0]*full.ndof
    i = 0

    # run time iterations
    for n in range(5001):
        upred = full.predict(mu, uprev)

        sigma = full.residual(mu, uprev, upred)
        X[:,i] = upred
        Y[:,i] = sigma
        i += 1

        # this is training, so we have the correct residual
        ucorr = full.correct(mu, uprev, zeros)
        uprev = ucorr
        mu['t'] += mu['dt']

    # save results to file
    np.save(f'test/X_a{int(alpha*10)}.npy', X.T)
    np.save(f'test/Y_a{int(alpha*10)}.npy', Y.T)
