import IFEM_CoSTA
import numpy as np
import matplotlib.pyplot as plt
import splipy as sp
import sys
from os import mkdir, path, listdir
import sys, os
from tqdm import tqdm
from contextlib import contextmanager


### Hack to supress stdout from IFEM

def fileno(file_or_fd):
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd

@contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    if stdout is None:
       stdout = sys.stdout

    stdout_fd = fileno(stdout)
    # copy stdout_fd before it is overwritten
    #NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied:
        stdout.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            #NOTE: dup2 makes stdout_fd inheritable unconditionally
            stdout.flush()
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied


def main():
    # split alpha values into disjoint test, train and validation sets
    train      = [    .1, .2, .3, .4, .5, .6,         .9, 1.0,      1.2, 1.3, 1.4,      1.6, 1.7, 1.8, 1.9, 2.0       ]
    validation = [                                .8,          1.1                                                    ]
    test       = [-.5,                        .7,                                  1.5,                          2.5  ]

    for alpha_values, set_name in zip([ train,   validation,   test],
                                      ['train', 'validation', 'test']):

        print(f'Creating {set_name} dataset')

        # create file structure test/train with all input files in it
        if not path.isdir(set_name):
            mkdir(set_name)

        # initialize the simulation
        ndof = 20
        nt   = 5000
        X = np.zeros((len(alpha_values)*nt, ndof))
        Y = np.zeros((len(alpha_values)*nt, ndof))
        Z = np.zeros((len(alpha_values)*nt, ndof))
        i = 0
        for alpha in tqdm(alpha_values):

            # set up physics based model
            with stdout_redirected():
                exact = IFEM_CoSTA.HeatEquation('P1.xinp')
                pbm   = IFEM_CoSTA.HeatEquation('pbm.xinp') # source term removed wrt above
                mu = {'dt':0.001, 't':0.0, 'ALPHA':alpha}
                u_ex_prev = exact.anasol(mu)['primary']
                u_h_prev  = pbm.initial_condition(mu)

            # run time iterations
            for n in tqdm(range(nt), leave=False):
                mu['t'] += mu['dt']

                with stdout_redirected():
                    u_ex_next = exact.anasol(mu)['primary']
                    u_h_next  = pbm.predict(mu, u_h_prev)
                    sigma     = pbm.residual(mu, u_ex_prev, u_ex_next)

                X[i,:] = u_h_next
                Y[i,:] = sigma
                Z[i,:] = u_ex_next
                i += 1

                u_ex_prev = u_ex_next
                u_h_prev  = u_h_next

        # save results to file
        np.save(f'{set_name}/X.npy', X)
        np.save(f'{set_name}/Y.npy', Y)
        np.save(f'{set_name}/Z.npy', Z)

    return


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

if __name__ == ('__main__'):
    main()

