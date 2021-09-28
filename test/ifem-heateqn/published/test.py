import IFEM_CoSTA
import splipy as sp
from keras.models import load_model
from os import listdir
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from contextlib import contextmanager
import sys, os

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

example_name = sys.argv[1]

ham = load_model(f'{example_name}/ham.h5')
ddm = load_model(f'{example_name}/ddm.h5')

spline = sp.Curve()
# spline.raise_order(1)
# spline.refine(17) # for a grand total of 20 controlpoints with p=2
spline.refine(18) # for a grand total of 20 controlpoints with p=1
spline = spline.set_dimension(1)


alphas = [-.5, .7, 1.5, 2.5  ]
for alpha in alphas:

    with stdout_redirected():
        pbm   = IFEM_CoSTA.HeatEquation(f'{example_name}/pbm.xinp')
        exact = IFEM_CoSTA.HeatEquation(f'{example_name}/{example_name}.xinp')
        mu    = {'dt':0.001, 't':0.0, 'ALPHA':alpha}
        u_pbm_prev = pbm.initial_condition(mu)
        u_ddm_prev = pbm.initial_condition(mu)
        u_ham_prev = pbm.initial_condition(mu)
        u_exact_prev = np.array(exact.initial_condition(mu))
    sigma = np.zeros(pbm.ndof)
    zeros = np.zeros(pbm.ndof)

    # accumulate error as a function of time steps
    pbm_err = np.zeros(5000)
    ham_err = np.zeros(5000)
    ddm_err = np.zeros(5000)
    
    # reshape data to fit into tensorflow
    u_ddm_prev = np.array(u_ddm_prev, ndmin=2)
                                                                               
    # run time iterations
    for n in tqdm(range(5000)):
        # update values
        mu['t'] += mu['dt']
        
        with stdout_redirected():
            # fetch the boundary conditions
            ud = pbm.initial_condition(mu)
            
            # create a predictor step for ham
            u_ham_pred = pbm.predict(mu, u_ham_prev)

            # create a correction term for ham
            sigma = ham.predict(np.array(u_ham_pred, ndmin=2)).flatten()
                                              
            # update the next step with a corrector
            u_exact_prev[:]  = exact.anasol(mu)['primary']
            u_pbm_prev[:]    = pbm.correct(mu, u_pbm_prev, zeros)
            u_ham_prev[:]    = pbm.correct(mu, u_ham_prev, sigma)
            u_ddm_prev       = ddm.predict(u_ddm_prev)
            u_ddm_prev[0,0]  = ud[0]
            u_ddm_prev[0,-1] = ud[-1]

        # log error
        pbm_err[n] = np.linalg.norm(u_pbm_prev           - u_exact_prev) / np.linalg.norm(u_exact_prev)
        ham_err[n] = np.linalg.norm(u_ham_prev           - u_exact_prev) / np.linalg.norm(u_exact_prev)
        ddm_err[n] = np.linalg.norm(u_ddm_prev.flatten() - u_exact_prev) / np.linalg.norm(u_exact_prev)

        
    plt.figure(figsize=(12,6))
    
    plt.plot(ham_err, 'g-')
    plt.plot(ddm_err, 'r-')
    plt.plot(pbm_err, 'b-')
    plt.legend(['exact', 'HAM', 'DDM', 'PBM'])
    plt.title(f'Relative errors for alpha={alpha}')
    plt.xlabel('Time level')
    plt.ylabel('Relative L2-error')
    plt.savefig(f'{example_name}/test/{example_name}_error_a{int(alpha*10)}.png')

        
    plt.figure(figsize=(12,6))
    
    x = np.linspace(0,1,20)
    N = spline.bases[0](x)
    plt.plot(x,(N@u_exact_prev).T, 'k-', mfc='none')
    plt.plot(x,(N@u_ham_prev).T, 'gd ', mfc='none')
    plt.plot(x,(N@u_ddm_prev.flatten()).T, 'ro ', mfc='none')
    plt.plot(x,(N@u_pbm_prev).T, 'bs ', mfc='none')
    plt.legend(['exact', 'HAM', 'DDM', 'PBM'])
    plt.xlabel('x [m]')
    plt.xlabel('T [C]')
    plt.title(f'Solution for alpha={alpha} n=5000')
    plt.savefig(f'{example_name}/test/{example_name}_final_sol_a{int(alpha*10)}.png')
