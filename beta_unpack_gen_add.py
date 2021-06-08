import numpy as np

def alpha(beta, knn_nn, knn_nb):
    knn_alpha=np.zeros((knn_nn*knn_nb,1))
    for i in range (0,knn_nn*knn_nb):
        knn_alpha[i]=beta[i]
        
    knn_alpha=np.reshape(knn_alpha,[knn_nn, knn_nb],order='F')
    return knn_alpha

def beta1(beta, knn_nn, knn_nb,data_out_dim):
    knn_beta1=np.zeros((data_out_dim,1))
    for i in range (0,data_out_dim):
        knn_beta1[i]=beta[i+knn_nn*knn_nb]
    return knn_beta1

def beta2(beta,knn_nn, knn_nb,data_out_dim):
    knn_beta0=np.zeros((data_out_dim,1))
    for i in range (0,data_out_dim):
        knn_beta0[i]=beta[knn_nn*knn_nb+data_out_dim+i]
    return knn_beta0

def rho(beta, knn_nn, knn_nb, data_out_dim):
    knn_rho=np.zeros((data_out_dim*knn_nn,1))
    for i in range (0,data_out_dim*knn_nn):
        knn_rho[i]=beta[knn_nn*knn_nb+2*data_out_dim+i]
    knn_rho=np.reshape(knn_rho,[knn_nn, data_out_dim],order='F')
    return knn_rho

def theta(beta, knn_nn, knn_nb, data_out_dim):
    knn_theta=np.zeros((knn_nn*knn_nb,1))
    for i in range (0,knn_nn*knn_nb):
        knn_theta[i]=beta[knn_nn*knn_nb+2*data_out_dim+data_out_dim*knn_nn+i]
    knn_theta=np.reshape(knn_theta,[knn_nn,knn_nb],order='F')
    return knn_theta

def v(beta,knn_nn, knn_nb, data_out_dim,data_in_dim):
    knn_v=np.zeros((knn_nb*data_in_dim,1))
    for i in range (0,knn_nb*data_in_dim):
        knn_v[i]=beta[2*knn_nn*knn_nb+2*data_out_dim+data_out_dim*knn_nn+i]
    knn_v=np.reshape(knn_v,[data_in_dim, knn_nb], order='F')
    return knn_v
"""
MATLAB script
knn.beta1=[beta(knn.nn*knn.nb+1:knn.nn*knn.nb+data.out_dim)]; %got to here 
knn.beta0=[beta(knn.nn*knn.nb+data.out_dim+1:knn.nn*knn.nb+2*data.out_dim)]; 
 
knn.rho=beta(knn.nn*knn.nb+2*data.out_dim+1:knn.nn*knn.nb+2*data.out_dim+data.out_dim*knn.nn);
knn.rho=reshape(knn.rho, [knn.nn data.out_dim]); 

knn.theta=beta(knn.nn*knn.nb+2*data.out_dim+1+data.out_dim*knn.nn:2*knn.nn*knn.nb+2*data.out_dim+data.out_dim*knn.nn);
knn.theta=reshape(knn.theta,[knn.nn knn.nb]);

knn.v=beta(2*knn.nn*knn.nb+2*data.out_dim+data.out_dim*knn.nn+1:end); 
knn.v=reshape(knn.v,[data.in_dim knn.nb]);  
"""