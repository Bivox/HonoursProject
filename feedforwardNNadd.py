import numpy as np 
import sigmoid 

def ff(knn_nn,knn_nb,knn_v,knn_alpha,knn_theta,knn_beta1,knn_beta0,Fc_train,x,\
       knn_rho,data_out_dim):    
    knn_rb=np.zeros((knn_nb,1))
    knn_r=np.zeros((knn_nn,2))
    
    for i in range(0,knn_nb):
        knn_rb[i]=np.dot(x,knn_v[:,i])
        
    for i in range(0,knn_nn):
        for j in range(0,knn_nb):
            if j==0:
                knn_r[i,0]=sigmoid.sigmoid(knn_alpha[i,j]*knn_rb[j]+knn_theta[i,j])
            else:
                knn_r[i,0]=knn_r[i,0]*sigmoid.sigmoid(knn_alpha[i,j]*knn_rb[j]+knn_theta[i,j])

    #normalising layer 
    normconst=np.sum(knn_r[:,0])
    for i in range(0,knn_nn):
        knn_r[i,1]=knn_r[i,0]/normconst
        
    #sum up rho times normalising layer 
    sumRho=np.zeros((1,data_out_dim))
    for i in range(0,knn_nn):
        sumRho=sumRho+np.dot(knn_rho[i,:],knn_r[i,1])
        
    #output 
    y=knn_beta1*Fc_train+sumRho+knn_beta0
    return y
###############################################################################
def rlayer(knn_nn,knn_nb,knn_v,knn_alpha,knn_theta,knn_beta1,knn_beta0,Fc_train,x,\
       knn_rho,data_out_dim):    
    knn_rb=np.zeros((knn_nb,1))
    knn_r=np.zeros((knn_nn,2))
    
    for i in range(0,knn_nb):
        knn_rb[i]=np.dot(x,knn_v[:,i])
        
    for i in range(0,knn_nn):
        for j in range(0,knn_nb):
            if j==0:
                knn_r[i,0]=sigmoid.sigmoid(knn_alpha[i,j]*knn_rb[j]+knn_theta[i,j])
            else:
                knn_r[i,0]=knn_r[i,0]*sigmoid.sigmoid(knn_alpha[i,j]*knn_rb[j]+knn_theta[i,j])
    #normalising layer 
    normconst=np.sum(knn_r[:,0])
    for i in range(0,knn_nn):
        knn_r[i,1]=knn_r[i,0]/normconst
    return knn_r
##############################################################################
def rblayer(knn_nn,knn_nb,knn_v,knn_alpha,knn_theta,knn_beta1,knn_beta0,Fc_train,x,\
       knn_rho,data_out_dim):    
    knn_rb=np.zeros((knn_nb,1))
    
    for i in range(0,knn_nb):
        knn_rb[i]=np.dot(x,knn_v[:,i])
        return knn_rb
