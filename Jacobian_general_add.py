import numpy as np
import sigmoid
import feedforwardNNadd as ffNN  

def Jacobian_gen(knn_nn,knn_nb,knn_v,knn_alpha,knn_theta,knn_beta1,knn_beta0,Fc_train,x,\
       knn_rho,data_out_dim,data_in_dim,Ye):  
    
    #feedforward pass
    y_p=ffNN.ff(knn_nn,knn_nb,knn_v,knn_alpha,knn_theta,knn_beta1,knn_beta0,Fc_train,x,\
       knn_rho,data_out_dim)
        
    knn_r=ffNN.rlayer(knn_nn,knn_nb,knn_v,knn_alpha,knn_theta,knn_beta1,knn_beta0,Fc_train,x,\
       knn_rho,data_out_dim)
        
    knn_rb=ffNN.rblayer(knn_nn,knn_nb,knn_v,knn_alpha,knn_theta,knn_beta1,knn_beta0,Fc_train,x,\
       knn_rho,data_out_dim)

    #evaluate gradients 
    grads_deps_dy=y_p-Ye
    
    grads_deps_dbeta1=grads_deps_dy*Fc_train

    grads_deps_dbeta0=grads_deps_dy
    
    grads_deps_drho=np.zeros((knn_nn,data_out_dim))
    for i in range(0,knn_nn):
        for j in range(0,data_out_dim):
            grads_deps_drho[i,j]=grads_deps_dy[j]*knn_r[i,1]
    grads_deps_drho=np.reshape(grads_deps_drho,[knn_nn*data_out_dim, 1],order='F')
    
    grads_gr_p=np.zeros((knn_nn,1))
    for j in range(0,knn_nn):
        grads_gr_p[j]=np.sum(np.multiply(grads_deps_dy,np.transpose(knn_rho[j,:])))
    
    grads_gr=grads_gr_p/np.sum(knn_r[:,0])
    for k in range(0,knn_nn):
        for j in range(0,knn_nn):
            grads_gr[k]=grads_gr[k]-grads_gr_p[j]*knn_r[j,1]/(np.sum(knn_r[:,0])**2)
            
    grads_gb=np.zeros((knn_nb,1))
    for j in range(0,knn_nb):
        for k in range(0,knn_nn):
            grads_gb[j]=grads_gb[j]+grads_gr[k]*knn_r[k,0]*\
                (1-sigmoid.sigmoid(knn_alpha[k,j]*knn_rb[j]+knn_theta[k,j]))*knn_alpha[k,j]

    grads_deps_dalpha=np.zeros((knn_nn,knn_nb))
    grads_deps_dtheta=np.zeros((knn_nn,knn_nb))
    
    for j in range(0,knn_nn):
        for k in range(0,knn_nb):
             grads_deps_dalpha[j,k]=grads_gr[j]*knn_r[j,0]*\
                 (1-sigmoid.sigmoid(knn_alpha[j,k]*knn_rb[k]+knn_theta[j,k]))*knn_rb[k] 
             grads_deps_dtheta[j,k]=grads_gr[j]*knn_r[j,0]*\
                 (1-sigmoid.sigmoid(knn_alpha[j,k]*knn_rb[k]+knn_theta[j,k]))
    grads_deps_dalpha=np.reshape(grads_deps_dalpha,[knn_nn*knn_nb,1],order='F')
    grads_deps_dtheta=np.reshape(grads_deps_dtheta,[knn_nn*knn_nb,1],order='F')
    
    grads_deps_dv=np.zeros((knn_nb,1))
    for i in range(0,knn_nb):
        for j in range(0,data_in_dim):
            grads_deps_dv[data_in_dim*i+j,0]=grads_gb[i]*x#[j]
    
    #Assemble Jacobian vector 
    J=np.concatenate((grads_deps_dalpha, grads_deps_dbeta1, grads_deps_dbeta0,\
                     grads_deps_drho, grads_deps_dtheta, grads_deps_dv),axis=0)
    return J