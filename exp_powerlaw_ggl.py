"""
author: Fabian Schaipp

Sigma denotes the covariance matrix, Theta the precision matrix
"""
from time import time
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.covariance import GraphicalLasso

from gglasso.solver.admm_solver import ADMM_MGL
from gglasso.solver.ggl_solver import PPDNA, warmPPDNA
from gglasso.helper.data_generation import group_power_network, sample_covariance_matrix, plot_degree_distribution
from gglasso.helper.experiment_helper import lambda_parametrizer, lambda_grid, discovery_rate, aic, ebic, error, draw_group_heatmap, get_default_plot_aes


p = 100
K = 5
N = 5000
N_train = 5000
M = 5

reg = 'GGL'

Sigma, Theta = group_power_network(p, K, M)

draw_group_heatmap(Theta)
#plot_degree_distribution(Theta)

S, sample = sample_covariance_matrix(Sigma, N)
S_train, sample_train = sample_covariance_matrix(Sigma, N_train)
Sinv = np.linalg.pinv(S, hermitian = True)

#%%
# grid search for best lambda values with warm starts
L1, L2, W2 = lambda_grid(num1 = 3, num2 = 9, reg = reg)
grid1 = L1.shape[0]; grid2 = L2.shape[1]

ERR = np.zeros((grid1, grid2))
FPR = np.zeros((grid1, grid2))
TPR = np.zeros((grid1, grid2))
DFPR = np.zeros((grid1, grid2))
DTPR = np.zeros((grid1, grid2))
AIC = np.zeros((grid1, grid2))
BIC = np.zeros((grid1, grid2))

FPR_GL = np.zeros((grid1, grid2))
TPR_GL = np.zeros((grid1, grid2))
DFPR_GL = np.zeros((grid1, grid2))
DTPR_GL = np.zeros((grid1, grid2))

Omega_0 = np.zeros((K,p,p))
Theta_0 = np.zeros((K,p,p))

for g1 in np.arange(grid1):
    for g2 in np.arange(grid2):
        lambda1 = L1[g1,g2]
        lambda2 = L2[g1,g2]
        
        # first solve single GLASSO
        singleGL = GraphicalLasso(alpha = lambda1 + lambda2/np.sqrt(2), tol = 1e-6, max_iter = 200, verbose = False)
        singleGL_sol = np.zeros((K,p,p))
        for k in np.arange(K):
            #model = quic.fit(S[k,:,:], verbose = 1)
            model = singleGL.fit(sample[k,:,:].T)
            singleGL_sol[k,:,:] = model.precision_
        
        TPR_GL[g1,g2] = discovery_rate(singleGL_sol, Theta)['TPR']
        FPR_GL[g1,g2] = discovery_rate(singleGL_sol, Theta)['FPR']
        DTPR_GL[g1,g2] = discovery_rate(singleGL_sol, Theta)['TPR_DIFF']
        DFPR_GL[g1,g2] = discovery_rate(singleGL_sol, Theta)['FPR_DIFF']
    
        
        #sol, info = PPDNA(S, lambda1, lambda2, reg, Omega_0, Theta_0 = Theta_0, sigma_0 = 10, max_iter = 20, \
        #                                            eps_ppdna = 1e-2 , verbose = False)
        
        sol, info = warmPPDNA(S_train, lambda1, lambda2, reg, Omega_0, Theta_0 = Theta_0, eps = 1e-3, verbose = False, measure = False)
        Theta_sol = sol['Theta']
        Omega_sol = sol['Omega']
        
        # warm start
        Omega_0 = Omega_sol.copy()
        Theta_0 = Theta_sol.copy()
        
        TPR[g1,g2] = discovery_rate(Theta_sol, Theta)['TPR']
        FPR[g1,g2] = discovery_rate(Theta_sol, Theta)['FPR']
        DTPR[g1,g2] = discovery_rate(Theta_sol, Theta)['TPR_DIFF']
        DFPR[g1,g2] = discovery_rate(Theta_sol, Theta)['FPR_DIFF']
        ERR[g1,g2] = error(Theta_sol, Theta)
        AIC[g1,g2] = aic(S_train, Theta_sol, N_train)
        BIC[g1,g2] = ebic(S_train, Theta_sol, N_train, gamma = 0.1)

# get optimal lambda
ix= np.unravel_index(BIC.argmin(), BIC.shape)
ix2= np.unravel_index(AIC.argmin(), AIC.shape)

#%%
l1opt = L1[ix]
l2opt = L2[ix]

print("Optimal lambda values: (l1,l2) = ", (l1opt,l2opt))

singleGL = GraphicalLasso(alpha = 3*l1opt, tol = 1e-6, max_iter = 200, verbose = True)
singleGL_sol = np.zeros((K,p,p))

for k in np.arange(K):
    #model = quic.fit(S[k,:,:], verbose = 1)
    model = singleGL.fit(sample[k,:,:].T)
    
    singleGL_sol[k,:,:] = model.precision_
    
discovery_rate(singleGL_sol, Theta)

draw_group_heatmap(Theta_sol, method = 'GLASSO')
#%%
Omega_0 = np.zeros((K,p,p))
Theta_0 = np.zeros((K,p,p))

sol, info = warmPPDNA(S, l1opt, l2opt, reg, Omega_0, Theta_0 = Theta_0, eps = 1e-5 , verbose = True)

#sol, info = ADMM_MGL(S, l1opt, l2opt, reg , Omega_0 , Theta_0 = Theta_0, rho = 1, max_iter = 100, eps_admm = 1e-5, \
#                     verbose = True, measure = False)

Theta_sol = sol['Theta']
Omega_sol = sol['Omega']

with sns.axes_style("white"):
    fig,axs = plt.subplots(nrows = 1, ncols = 3, figsize = (10,3))
    draw_group_heatmap(Theta, method = 'truth', ax = axs[0])
    draw_group_heatmap(Theta_sol, method = 'PPDNA', ax = axs[1])
    draw_group_heatmap(singleGL_sol, method = 'GLASSO', ax = axs[2])

#%%
def plot_fpr_tpr(FPR, TPR, W2, ix, ix2):
    """
    plots the FPR vs. TPR pathes
    ix and ix2 are the lambda values with optimal eBIC/AIC
    """
    plot_aes = get_default_plot_aes()

    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots(1,1)
        ax.plot(FPR.T, TPR.T, **plot_aes)
        #ax.plot(FPR_GL.T, TPR_GL.T, c = 'grey', **plot_aes)
        
        ax.plot(FPR[ix], TPR[ix], marker = 'o', fillstyle = 'none', markersize = 20, markeredgecolor = '#12cf90')
        ax.plot(FPR[ix2], TPR[ix2], marker = 'o', fillstyle = 'none', markersize = 20, markeredgecolor = '#cf3112')
    
        ax.set_xlim(-.01,1)
        ax.set_ylim(-.01,1)
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(labels = [f"w2 = {w}" for w in W2], loc = 'lower right')
        
    fig.suptitle('Discovery rate for different regularization strengths')
    
    return

def plot_diff_fpr_tpr(DFPR, DTPR, DFPR_GL, DTPR_GL, W2, ix, ix2):
    """
    plots the FPR vs. TPR pathes 
    _GL indicates the solution of single Graphical Lasso
    ix and ix2 are the lambda values with optimal eBIC/AIC
    """
    plot_aes = get_default_plot_aes()

    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots(1,1)
        ax.plot(DFPR.T, DTPR.T, **plot_aes)
        ax.plot(DFPR_GL.T, DTPR_GL.T, c = 'grey', **plot_aes)
        
        ax.plot(DFPR[ix], DTPR[ix], marker = 'o', fillstyle = 'none', markersize = 20, markeredgecolor = '#12cf90')
        ax.plot(DFPR[ix2], DTPR[ix2], marker = 'o', fillstyle = 'none', markersize = 20, markeredgecolor = '#cf3112')
    
        ax.set_xlim(-.01,1)
        ax.set_ylim(-.01,1)
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(labels = [f"w2 = {w}" for w in W2], loc = 'lower right')
        
    fig.suptitle('Discovery rate for different regularization strengths')
    
    return

plot_fpr_tpr(FPR, TPR, W2, ix, ix2)
plot_diff_fpr_tpr(DFPR, DTPR, DFPR_GL, DTPR_GL, W2, ix, ix2)
#%%
# accuracy impact on total error analysis
#L1, L2 = lambda_grid(num1 = 1, num2 = 6, reg = reg)

L2 = l2opt*np.linspace(-.5,.5,5) + l2opt
L1 = lambda_parametrizer(L2, w2 = 0.35)
grid1 = L1.shape[0]

grid2 = 5
EPS = np.logspace(start = -.5, stop = -5, num = grid2, base = 10)
EPS = np.array([2e-1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5])

grid2 = len(EPS)

ERR = np.zeros((grid1, grid2))
AIC = np.zeros((grid1, grid2))
RT = np.zeros((grid1, grid2))

#Omega_0 = np.zeros((K,p,p)); Theta_0 = np.zeros((K,p,p)); X_0 = np.zeros((K,p,p))
Om_0_0 = np.zeros((K,p,p)); Th_0_0 = np.zeros((K,p,p)); X_0_0 = np.zeros((K,p,p))

for g1 in np.arange(grid1):
    
    Omega_0 = Om_0_0.copy()
    Theta_0 = Th_0_0.copy()
    X_0 = X_0_0.copy()
    
    for g2 in np.arange(grid2):
            
        start = time()
        #sol, info = PPDNA(S, L1[g1], L2[g1], reg, Omega_0, Theta_0 = Theta_0, X_0 = X_0, \
        #                                                eps_ppdna = EPS[g2] , verbose = False)
        sol, info = ADMM_MGL(S, L1[g1], L2[g1], reg , Omega_0 , Theta_0 = Theta_0, X_0 = X_0, rho = 1, max_iter = 100, \
                             eps_admm = EPS[g2], verbose = False)
        end = time()
        
        Theta_sol = sol['Theta']
        Omega_sol = sol['Omega']
        X_sol = sol['X']
        
        # warm start, has to be resetted whenever changing g1 
        Omega_0 = Omega_sol.copy(); Theta_0 = Theta_sol.copy(); X_0 = X_sol.copy()
        if g2 == 0:
            Om_0_0 = Omega_sol.copy(); Th_0_0 = Theta_sol.copy(); X_0_0 = X_sol.copy()
        
        ERR[g1,g2] = error(Theta_sol, Theta)
        AIC[g1,g2] = aic(S,Theta_sol,N)
        RT[g1,g2] = end-start


#%%

pal = sns.color_palette("GnBu_d", grid1)
plot_aes = get_default_plot_aes()

with sns.axes_style("whitegrid"):
    fig, ax = plt.subplots(1,1)
    for l in np.arange(grid1):
        ax.plot(EPS, ERR[l,:], c=pal[l],**plot_aes )

    ax.set_xlim(EPS.max()*2 , EPS.min()/2)
    ax.set_ylim(0,0.3)
    ax.set_xscale('log')
    
    ax.set_xlabel('Solution accuracy')
    ax.set_ylabel('Total relative error')
    ax.legend(labels = ["l2 = " + "{:.2E}".format(l) for l in L2])
    
fig.suptitle('Total error for different solution accuracies')


#%%

f = np.array([0.5,0.9, 1.5, 5, 10, 20])
N = (f * p).astype(int)

l1 = 1e-2 * np.ones(len(f))
l2 = 1e-2 * np.ones(len(f))

#mapf = lambda x: .1 + 1/(np.log(x+1))

#l1 = mapf(f)*l1opt
#l2 = mapf(f)*l2opt

Omega_0 = np.zeros((K,p,p))

RT_ADMM = np.zeros(len(N))
RT_PPA = np.zeros(len(N))
TPR = np.zeros(len(N))
FPR = np.zeros(len(N))


for j in np.arange(len(N)):
    
    #Omega_0 = np.linalg.pinv(S, hermitian = True)
    S, sample = sample_covariance_matrix(Sigma, N[j])
    
    start = time()
    sol, info = ADMM_MGL(S, l1[j], l2[j], reg , Omega_0 , rho = 1, max_iter = 1000, \
                             eps_admm = 5e-5, verbose = False, measure = True)
    
    end = time()
    
    RT_ADMM[j] = end-start
    TPR[j] = discovery_rate(sol['Theta'], Theta)['TPR']
    FPR[j] = discovery_rate(sol['Theta'], Theta)['FPR']
    
    start = time()
    sol, info = warmPPDNA(S, l1[j], l2[j], reg, Omega_0, eps = 5e-5, verbose = False, measure = True)
    end = time()
    
    RT_PPA[j] = end-start
    
    
plt.plot(RT_ADMM)
plt.plot(RT_PPA)