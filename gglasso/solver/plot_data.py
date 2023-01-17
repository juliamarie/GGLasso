#import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt


#
# ############################################## penalty sensibility ################################################

data_admm_1 = genfromtxt('/Users/julia/Documents/Uni/Bachelor_Arbeit/code/GGLasso/gglasso/tests/data_for_plot_admm_pensens1.csv',
                       delimiter=',')
data_admm_2 = genfromtxt('/Users/julia/Documents/Uni/Bachelor_Arbeit/code/GGLasso/gglasso/tests/data_for_plot_admm_pensens2.csv',
                        delimiter=',')


x_axis = range(1, 1001)
plt.plot(x_axis, abs(data_admm_1[0]+18.9493947), 'navy', label=r'$\rho=0.01$')
plt.plot(x_axis, abs(data_admm_1[2]+18.9493947), 'royalblue', label=r'$\rho=0.1$')
plt.plot(x_axis, abs(data_admm_1[4]+18.9493947), 'deepskyblue', label=r'$\rho=1$')
plt.plot(x_axis, abs(data_admm_1[6]+18.9493947), 'lightskyblue', label=r'$\rho=2$')
#plt.title(r'Sensibility of the penalty parameter, $\lambda_1=0.1$ and $\lambda_2=0.1$')
plt.yscale('log')
plt.xlabel("number of iterations")
plt.ylabel("distance to the optimal value")
plt.legend()
plt.savefig('/Users/julia/Documents/Uni/Bachelor_Arbeit/Bachelor_Arbeit_File/pensens1_o.pdf')
plt.show()


x_axis = range(1, 1001)
plt.plot(x_axis, data_admm_1[1], 'navy', label=r'$\rho=0.01$')
plt.plot(x_axis, data_admm_1[3], 'royalblue', label=r'$\rho=0.1$')
plt.plot(x_axis, data_admm_1[5], 'deepskyblue', label=r'$\rho=1$')
plt.plot(x_axis, data_admm_1[7], 'lightskyblue', label=r'$\rho=2$')
plt.yscale('log')
#plt.title(r'Sensibility of the penalty parameter, $\lambda_1=0.1$ and $\lambda_2=0.1$')
plt.xlabel("number of iterations")
plt.ylabel("feasibility")
plt.legend()
plt.savefig('/Users/julia/Documents/Uni/Bachelor_Arbeit/Bachelor_Arbeit_File/pensens1_f.pdf')
plt.show()

x_axis = range(1, 1001)
plt.plot(x_axis, abs(data_admm_2[0]+9.33350925), 'navy', label=r'$\rho=0.01$')
plt.plot(x_axis, abs(data_admm_2[2]+9.33350925), 'royalblue', label=r'$\rho=0.1$')
plt.plot(x_axis, abs(data_admm_2[4]+9.33350925), 'deepskyblue', label=r'$\rho=1$')
plt.plot(x_axis, abs(data_admm_2[6]+9.33350925), 'lightskyblue', label=r'$\rho=2$')
#plt.title(r'Sensibility of the penalty parameter, $\lambda_1=0.2$ and $\lambda_2=0.5$')
plt.yscale('log')
plt.xlabel("number of iterations")
plt.ylabel("distance to the optimal value")
plt.legend()
plt.savefig('/Users/julia/Documents/Uni/Bachelor_Arbeit/Bachelor_Arbeit_File/pensens2_o.pdf')
plt.show()


x_axis = range(1, 1001)
plt.plot(x_axis, data_admm_2[1], 'navy', label=r'$\rho=0.01$')
plt.plot(x_axis, data_admm_2[3], 'royalblue', label=r'$\rho=0.1$')
plt.plot(x_axis, data_admm_2[5], 'deepskyblue', label=r'$\rho=1$')
plt.plot(x_axis, data_admm_2[7], 'lightskyblue', label=r'$\rho=2$')
plt.yscale('log')
#plt.title(r'Sensibility of the penalty parameter, $\lambda_1=0.2$ and $\lambda_2=0.5$')
plt.xlabel("number of iterations")
plt.ylabel("feasibility")
plt.legend()
plt.savefig('/Users/julia/Documents/Uni/Bachelor_Arbeit/Bachelor_Arbeit_File/pensens2_f.pdf')
plt.show()
#




################################################ ADMM vs LAADMM  #################################################
data_admm1 = genfromtxt('/Users/julia/Documents/Uni/Bachelor_Arbeit/code/GGLasso/gglasso/tests/data_for_plot_admm_pensens1.csv',
                       delimiter=',')
data_admm2 = genfromtxt('/Users/julia/Documents/Uni/Bachelor_Arbeit/code/GGLasso/gglasso/tests/data_for_plot_admm_pensens2.csv',
                       delimiter=',')
data_la_admm1 = genfromtxt(
    '/Users/julia/Documents/Uni/Bachelor_Arbeit/code/GGLasso/gglasso/tests/data_for_plot_la_admm_pensens1.csv', delimiter=',')
data_la_admm2 = genfromtxt(
    '/Users/julia/Documents/Uni/Bachelor_Arbeit/code/GGLasso/gglasso/tests/data_for_plot_la_admm_pensens2.csv', delimiter=',')
x_axis = range(1, 1001)


plt.plot(x_axis, abs(data_admm1[0]+18.9493947), 'navy', label=r'ADMM, $\rho=0.01$')
plt.plot(x_axis, abs(data_admm1[2]+18.9493947), 'royalblue', label=r'ADMM, $\rho=0.1$')
plt.plot(x_axis, abs(data_admm1[4]+18.9493947), 'deepskyblue', label=r'ADMM, $\rho=1$')
plt.plot(x_axis, abs(data_admm1[6]+18.9493947), 'lightskyblue', label=r'ADMM, $\rho=2$')
plt.plot(x_axis, abs(data_la_admm1[0]+18.9493947), 'navy', linestyle='dashed', label=r'LA-ADMM, $\rho_1=0.01$')
plt.plot(x_axis, abs(data_la_admm1[2]+18.9493947), 'royalblue', linestyle='dashed', label=r'LA-ADMM, $\rho_1=0.1$')
plt.plot(x_axis, abs(data_la_admm1[4]+18.9493947), 'deepskyblue', linestyle='dashed', label=r'LA-ADMM, $\rho_1=1$')
plt.plot(x_axis, abs(data_la_admm1[6]+18.9493947), 'lightskyblue', linestyle='dashed', label=r'LA-ADMM, $\rho=2$')
plt.vlines([100,200,300,400,500,600,700,800,900],0,1,colors='k', linestyles='dashed')
#plt.plot(x_axis, data_admm[1], 'b--', label='ADMM: feasability')
#plt.plot(x_axis, data_la_admm[1], 'r--', label='LA-ADMM: feasability')
plt.yscale('log')
#plt.title(r'ADMM vs. LA-ADMM: $\rho = 1$')
plt.xlabel("number of iterations")
plt.ylabel("distance to the optimal value")
plt.legend()
plt.savefig('/Users/julia/Documents/Uni/Bachelor_Arbeit/Bachelor_Arbeit_File/Figure_1_o.pdf')
plt.show()

plt.plot(x_axis, data_admm1[1], 'navy', label=r'ADMM, $\rho=0.01$')
plt.plot(x_axis, data_admm1[3], 'royalblue', label=r'ADMM, $\rho=0.1$')
plt.plot(x_axis, data_admm1[5], 'deepskyblue', label=r'ADMM, $\rho=1$')
plt.plot(x_axis, data_admm1[7], 'lightskyblue', label=r'ADMM, $\rho=2$')
plt.plot(x_axis, data_la_admm1[1], 'navy', linestyle='dashed', label=r'LA-ADMM, $\rho_1=0.01$')
plt.plot(x_axis, data_la_admm1[3], 'royalblue', linestyle='dashed', label=r'LA-ADMM, $\rho_1=0.1$')
plt.plot(x_axis, data_la_admm1[5], 'deepskyblue', linestyle='dashed', label=r'LA-ADMM, $\rho_1=1$')
plt.plot(x_axis, data_la_admm1[7], 'lightskyblue', linestyle='dashed', label=r'LA-ADMM, $\rho=2$')
plt.vlines([100,200,300,400,500,600,700,800,900],0,1,colors='k', linestyles='dashed')
#plt.plot(x_axis, data_admm[1], 'b--', label='ADMM: feasability')
#plt.plot(x_axis, data_la_admm[1], 'r--', label='LA-ADMM: feasability')
plt.yscale('log')
#plt.title(r'ADMM vs. LA-ADMM: $\rho = 1$')
plt.xlabel("number of iterations")
plt.ylabel("feasibility")
plt.legend()
plt.savefig('/Users/julia/Documents/Uni/Bachelor_Arbeit/Bachelor_Arbeit_File/Figure_1_f.pdf')
plt.show()






plt.plot(x_axis, abs(data_admm2[0]+9.33350925), 'navy', label=r'ADMM, $\rho=0.01$')
plt.plot(x_axis, abs(data_admm2[2]+9.33350925), 'royalblue', label=r'ADMM, $\rho=0.1$')
plt.plot(x_axis, abs(data_admm2[4]+9.33350925), 'deepskyblue', label=r'ADMM, $\rho=1$')
plt.plot(x_axis, abs(data_admm2[6]+9.33350925), 'lightskyblue', label=r'ADMM, $\rho=2$')
plt.plot(x_axis, abs(data_la_admm2[0]+9.33350925), 'navy', linestyle='dashed', label=r'LA-ADMM, $\rho_1=0.01$')
plt.plot(x_axis, abs(data_la_admm2[2]+9.33350925), 'royalblue', linestyle='dashed', label=r'LA-ADMM, $\rho_1=0.1$')
plt.plot(x_axis, abs(data_la_admm2[4]+9.33350925), 'deepskyblue', linestyle='dashed', label=r'LA-ADMM, $\rho_1=1$')
plt.plot(x_axis, abs(data_la_admm2[6]+9.33350925), 'lightskyblue', linestyle='dashed', label=r'LA-ADMM, $\rho=2$')
plt.vlines([100,200,300,400,500,600,700,800,900],0,1,colors='k', linestyles='dashed')
#plt.plot(x_axis, data_admm[1], 'b--', label='ADMM: feasability')
#plt.plot(x_axis, data_la_admm[1], 'r--', label='LA-ADMM: feasability')
plt.yscale('log')
#plt.title(r'ADMM vs. LA-ADMM: $\rho = 1$')
plt.xlabel("number of iterations")
plt.ylabel("distance to the optimal value")
plt.legend()
plt.savefig('/Users/julia/Documents/Uni/Bachelor_Arbeit/Bachelor_Arbeit_File/Figure_2_o.pdf')
plt.show()

plt.plot(x_axis, data_admm2[1], 'navy', label=r'ADMM, $\rho=0.01$')
plt.plot(x_axis, data_admm2[3], 'royalblue', label=r'ADMM, $\rho=0.1$')
plt.plot(x_axis, data_admm2[5], 'deepskyblue', label=r'ADMM, $\rho=1$')
plt.plot(x_axis, data_admm2[7], 'lightskyblue', label=r'ADMM, $\rho=2$')
plt.plot(x_axis, data_la_admm2[1], 'navy', linestyle='dashed', label=r'LA-ADMM, $\rho_1=0.01$')
plt.plot(x_axis, data_la_admm2[3], 'royalblue', linestyle='dashed', label=r'LA-ADMM, $\rho_1=0.1$')
plt.plot(x_axis, data_la_admm2[5], 'deepskyblue', linestyle='dashed', label=r'LA-ADMM, $\rho_1=1$')
plt.plot(x_axis, data_la_admm2[7], 'lightskyblue', linestyle='dashed', label=r'LA-ADMM, $\rho=2$')
plt.vlines([100,200,300,400,500,600,700,800,900],0,1,colors='k', linestyles='dashed')
#plt.plot(x_axis, data_admm[1], 'b--', label='ADMM: feasability')
#plt.plot(x_axis, data_la_admm[1], 'r--', label='LA-ADMM: feasability')
plt.yscale('log')
#plt.title(r'ADMM vs. LA-ADMM: $\rho = 1$')
plt.xlabel("number of iterations")
plt.ylabel("feasibility")
plt.legend()
plt.savefig('/Users/julia/Documents/Uni/Bachelor_Arbeit/Bachelor_Arbeit_File/Figure_2_f.pdf')
plt.show()

