import IFEM_CoSTA
import splipy as sp
from keras.models import load_model
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import sys

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

    pbm   = IFEM_CoSTA.HeatEquation(f'{example_name}/pbm.xinp', verbose=False)
    mu    = {'dt':0.001, 't':0.0, 'ALPHA':alpha}
    u_pbm_prev = pbm.anasol(mu)['primary']
    u_ddm_prev = pbm.anasol(mu)['primary']
    u_ham_prev = pbm.anasol(mu)['primary']
    u_exact_prev = np.array(pbm.anasol(mu)['primary'])
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
        
        # fetch the boundary conditions
        ud = pbm.anasol(mu)['primary']

        # create a predictor step for ham
        u_ham_pred = pbm.predict(mu, u_ham_prev)

        # create a correction term for ham
        sigma = ham.predict(np.array(u_ham_pred, ndmin=2)).flatten()

        # update the next step with a corrector
        u_exact_prev[:]  = pbm.anasol(mu)['primary']
        u_pbm_prev[:]    = pbm.predict(mu, u_pbm_prev)
        u_ham_prev[:]    = pbm.correct(mu, u_ham_prev, sigma)
        u_ddm_prev       = ddm.predict(u_ddm_prev)
        u_ddm_prev[0,0]  = ud[0]
        u_ddm_prev[0,-1] = ud[-1]

        # log error
        pbm_err[n] = np.linalg.norm(u_pbm_prev           - u_exact_prev) / np.linalg.norm(u_exact_prev)
        ham_err[n] = np.linalg.norm(u_ham_prev           - u_exact_prev) / np.linalg.norm(u_exact_prev)
        ddm_err[n] = np.linalg.norm(u_ddm_prev.flatten() - u_exact_prev) / np.linalg.norm(u_exact_prev)


    plt.figure(figsize=(12,6))
    
    plt.semilogy(ham_err, 'g-')
    plt.semilogy(ddm_err, 'r-')
    plt.semilogy(pbm_err, 'b-')
    plt.legend(['HAM', 'DDM', 'PBM'])
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
