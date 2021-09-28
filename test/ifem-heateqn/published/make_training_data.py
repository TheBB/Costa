import IFEM_CoSTA
import numpy as np
import splipy as sp
from os import mkdir, path, listdir
import sys, os
from tqdm import tqdm
from contextlib import contextmanager


def main(*argv):
    example_name = argv[1]
    
    # split alpha values into disjoint test, train and validation sets
    train      = [    .1, .2, .3, .4, .5, .6,         .9, 1.0,      1.2, 1.3, 1.4,      1.6, 1.7, 1.8, 1.9, 2.0       ]
    validation = [                                .8,          1.1                                                    ]
    test       = [-.5,                        .7,                                  1.5,                          2.5  ]

    for alpha_values, set_name in zip([ train,   validation,   test],
                                      ['train', 'validation', 'test']):

        print(f'Creating {example_name}/{set_name} dataset')

        # create file structure test/train with all input files in it
        if not path.isdir(f'{example_name}/{set_name}'):
            mkdir(f'{example_name}/{set_name}')

        # initialize the simulation
        ndof = 20
        nt   = 5000
        X   = np.zeros((len(alpha_values)*nt, ndof))
        Y   = np.zeros((len(alpha_values)*nt, ndof))
        Xex = np.zeros((len(alpha_values)*nt, ndof))
        Yex = np.zeros((len(alpha_values)*nt, ndof))
        i = 0
        for alpha in tqdm(alpha_values):

            # set up physics based model
            pbm   = IFEM_CoSTA.HeatEquation(f'{example_name}/pbm.xinp', verbose=False) # source term removed wrt above
            mu = {'dt':0.001, 't':0.0, 'ALPHA':alpha}
            u_ex_prev = pbm.anasol(mu)['primary']

            # run time iterations
            for n in tqdm(range(nt), leave=False):
                mu['t'] += mu['dt']

                u_ex_next = pbm.anasol(mu)['primary']
                u_h_next  = pbm.predict(mu, u_ex_prev)
                sigma     = pbm.residual(mu, u_ex_prev, u_ex_next)

                X[i,:] = u_h_next
                Y[i,:] = sigma
                Xex[i,:] = u_ex_prev
                Yex[i,:] = u_ex_next
                i += 1

                u_ex_prev = u_ex_next

        # save results to file
        np.save(f'{example_name}/{set_name}/X.npy', X)
        np.save(f'{example_name}/{set_name}/Y.npy', Y)
        np.save(f'{example_name}/{set_name}/Xex.npy', Xex)
        np.save(f'{example_name}/{set_name}/Yex.npy', Yex)

    return


if __name__ == ('__main__'):
    main(*sys.argv)

