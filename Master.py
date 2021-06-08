import math
import numpy as np
import beta_unpack_gen_add as unpack
import Jacobian_general_add as Jacobian 
import feedforwardNNadd as ffNN  
import matplotlib.pyplot as plt
######################################################################
# Generate data
A=10 #amplitude of sine wave 
f=1000 #sine wave freq 
n=5 #no of cycles of data 
T=1/f #time period of sine wave 

t=[0]*(100*n+1) #time steps
sc=[0]*(100*n+1) #coarse data 
se=[0]*(100*n+1) #expensive data

for i in range(0,100*n+1):
    t[i]=i*T/100
    sc[i]=A*math.exp(-t[i]*1000)*math.sin(2*math.pi*f*t[i])
    se[i]=A*math.exp(-t[i]*900)*math.sin(2*math.pi*f*t[i]+math.pi/4)

data_norm=2*A
data_norm_t=max(t)
si=60 #sampling interval

####
#Select indices of expensive data points 
#xintercepts and peaks/troughs
data_idx=[1,	39,	39,	61,	89,	89,	139,	139,	161,	189,	189,\
          	239,	261,	289,	289,	339,	339,	361,	389,\
                  	439,	439,	461,	489,	489]
#data points for extrapolation problem (needs regularisation turned on)
#data_idx=[5, 10, 15, 20, 25]
####
    
#get scattered expensive data for plotting later 
te=np.zeros((len(data_idx),1))
fe_plot=np.zeros((len(data_idx),1))
for i in range (0,len(data_idx)):
    te[i]=t[data_idx[i]]
    fe_plot[i]=se[data_idx[i]]

#Normalise data for knn training 
data_Fc=[0]*(len(data_idx))
data_Fe=[0]*(len(data_idx))
data_t=[0]*(len(data_idx))
for i in range(0,len(data_idx)):
    data_Fc[i]=sc[data_idx[i]]/data_norm+0.5
    data_Fe[i]=se[data_idx[i]]/data_norm+0.5
    data_t[i]=t[data_idx[i]]/data_norm_t

data_tfull=t
data_Fcfull=sc
data_Fefull=se
data_in_dim=1
data_out_dim=1


######################################################################
#Define KNN geometry and initialise weights and biases
knn_nn=10 # of neurons in R and R'
knn_nb=10 # of neuron in the boundary layer B

knn_v=2*math.sqrt(1/knn_nn)*np.random.rand(data_in_dim,knn_nb)-1/math.sqrt(1/knn_nn)
knn_alpha=2*math.sqrt(1/knn_nn)*np.random.rand(knn_nn,knn_nb)-1/math.sqrt(1/knn_nn)
knn_theta=np.zeros((knn_nn,knn_nb)) #knn.theta=2*rand(knn.nn,knn.nb)-1;

knn_beta1=np.ones((1,data_out_dim))
knn_beta0=0.05*np.random.rand(1,data_out_dim)
knn_rho=np.random.rand(knn_nn,data_out_dim)

#######################################################################
#Arrange hyperparameters into vector, beta 
P=len(data_idx) # high fiedlity samples
y_p=np.zeros((P,1))  # vector of Knn predictions 

beta=np.concatenate((np.reshape(knn_alpha,[knn_nn*knn_nb,1],order='F'), knn_beta1, knn_beta0,\
np.reshape(knn_rho, [knn_nn*data_out_dim, 1], order='F'),\
np.reshape(knn_theta,[knn_nn*knn_nb,1], order='F'),np.reshape(knn_v,[data_in_dim*knn_nb,1])),axis=0)

beta_first=beta
nn_train_beta_store=np.concatenate((beta_first, beta_first+1),axis=1)

#########################################################################
#Backpropagation through sgd
nn_train_convergence=0          #convergence flag 
nn_train_iters=0                #iterations counter 
nn_train_Jacob_len=len(beta)    #total # hyperparameters 
nn_train_eps=0.025               #SGD learning rate
nn_train_lambda=0#1e-2          #Regularisation rate

nn_train_delta_old=np.ones((nn_train_Jacob_len,1))
#nn_train_deltaW=np.zeros(nn_train.Jacob_len,1)
nn_train_MSE_old=1e6
nn_train_G=np.zeros((nn_train_Jacob_len,nn_train_Jacob_len))


while nn_train_convergence==0:
    nn_train_iters=nn_train_iters+1
    
    #Rearrange order of training data
    x_old=data_t
    Ye_old=data_Fe 
    Yc_old=data_Fc
    nn_train_new_order=np.random.permutation(P)
    
    Parray=[0]*len(x_old)
    Ye_store=[0]*len(Ye_old)
    Yc_store=[0]*len(Yc_old)
    for i in range(0,P):
        Parray[i]=x_old[nn_train_new_order[i]]
        Ye_store[i]=Ye_old[nn_train_new_order[i]]
        Yc_store[i]=Yc_old[nn_train_new_order[i]]
        
    #Iterate through training data, adjust hyperparams 
    for k2 in range(0,P):
        x=[0]*data_in_dim
        x=Parray[k2]
        Ye=Ye_store[k2]
        Fc_train=Yc_store[k2]
        
        #unpack beta 
        knn_alpha=unpack.alpha(beta, knn_nn, knn_nb)
        knn_beta1=unpack.beta1(beta, knn_nn, knn_nb,data_out_dim)
        knn_beta0=unpack.beta2(beta,knn_nn, knn_nb,data_out_dim)
        knn_rho=unpack.rho(beta, knn_nn, knn_nb, data_out_dim)
        knn_theta=unpack.theta(beta, knn_nn, knn_nb, data_out_dim)
        knn_v=unpack.v(beta,knn_nn, knn_nb, data_out_dim,data_in_dim)
        
        Grad=Jacobian.Jacobian_gen(knn_nn,knn_nb,knn_v,knn_alpha,knn_theta,knn_beta1,knn_beta0,Fc_train,x,\
                                   knn_rho,data_out_dim,data_in_dim,Ye)
        
        #SGD- used adagrad variant 
        for i_ada in range(0,nn_train_Jacob_len):
            if k2==0:
                #Update prior gradients- note that I only do this using the gradient from the 
                # first data entry/iteration to stop this term from blowing up too quickly 
                nn_train_G[i_ada,i_ada]=nn_train_G[i_ada,i_ada]+Grad[i_ada]**2
            
            #Update the hyperparams (including regularisation)
            #Note that the regulrisation forces rhos-->1, other params to 0
            #Update is sensitive to the sign of the hyperparameters
            
            if beta[i_ada]<=0:
                if i_ada>=knn_nn*knn_nb and i_ada<=knn_nn*knn_nb+data_out_dim-1:
                    beta[i_ada]=beta[i_ada]-nn_train_eps*Grad[i_ada]/\
                        math.sqrt(nn_train_G[i_ada,i_ada]+1e-8)-1*nn_train_lambda*((2*(beta[i_ada]-1)))
                else:
                    beta[i_ada]=beta[i_ada]-nn_train_eps*Grad[i_ada]/\
                        math.sqrt(nn_train_G[i_ada,i_ada]+1e-8)+nn_train_lambda
                        
            if beta[i_ada]>0:
                if i_ada>=knn_nn*knn_nb and i_ada<=knn_nn*knn_nb+data_out_dim-1:
                    beta[i_ada]=beta[i_ada]-nn_train_eps*Grad[i_ada]/\
                        math.sqrt(nn_train_G[i_ada,i_ada]+1e-8)-1*nn_train_lambda*((2*(beta[i_ada]-1)))
                else:
                    beta[i_ada]=beta[i_ada]-nn_train_eps*Grad[i_ada]/\
                        math.sqrt(nn_train_G[i_ada,i_ada]+1e-8)-nn_train_lambda
     
    #Evaluate end of iteration MSE 
    knn_alpha=unpack.alpha(beta, knn_nn, knn_nb)
    knn_beta1=unpack.beta1(beta, knn_nn, knn_nb,data_out_dim)
    knn_beta0=unpack.beta2(beta,knn_nn, knn_nb,data_out_dim)
    knn_rho=unpack.rho(beta, knn_nn, knn_nb, data_out_dim)
    knn_theta=unpack.theta(beta, knn_nn, knn_nb, data_out_dim)
    knn_v=unpack.v(beta,knn_nn, knn_nb, data_out_dim,data_in_dim)
    rerror_tot=np.zeros((1,P))
    for k2 in range(0,P):
        x=[0]*data_in_dim
        Ye=[0]*data_out_dim
        x=Parray[k2]
        Ye=Ye_store[k2]
        Fc_train=Yc_store[k2]
        y_p=ffNN.ff(knn_nn,knn_nb,knn_v,knn_alpha,knn_theta,knn_beta1,knn_beta0,Fc_train,x,\
                    knn_rho,data_out_dim)
        rerror_tot[0,k2]=y_p-Ye
        
    MSE=0.5*np.sum(np.multiply(rerror_tot,rerror_tot))
    
    nn_train_beta_store[:,0]=nn_train_beta_store[:,1]
    
    for i in range(0,nn_train_Jacob_len):
        nn_train_beta_store[i,1]=beta[i]
    
    #####
    #plot as algorithm goes 
    if nn_train_iters % 200==0:
        #Plot the KNN response every 200 iterations 
        y_full=np.zeros((100*n+1,1))
        x=np.zeros((1,data_in_dim))
        for k2 in range(0,len(data_tfull)):
            Fc_train=(data_Fcfull[k2]/data_norm)+0.5
            x=data_tfull[k2]/data_norm_t
            y_full[k2]=ffNN.ff(knn_nn,knn_nb,knn_v,knn_alpha,knn_theta,knn_beta1,knn_beta0,Fc_train,x,\
                    knn_rho,data_out_dim)
            y_full[k2]=(y_full[k2]-0.5)*data_norm

        fig, ax = plt.subplots()
        ax.plot(t, y_full, 'g',label='KNN') 
        ax.plot(t,sc, 'r',label='Coarse data')
        ax.scatter(te,fe_plot,marker='+',label='Expensive data')
        ax.legend()
        ax.set_xlabel('t')
        ax.set_ylabel('f(t)') 
        ax.set_title('Iter no %d' %nn_train_iters)
        
        plt.pause(0.05)
    ###
        
    print("Iter no",nn_train_iters, " MSE error", MSE)
    
    if nn_train_iters>50 and np.linalg.norm(nn_train_beta_store[:,0]\
            -nn_train_beta_store[:,1])<1e-4:
        nn_train_convergence=1
        
    if nn_train_iters>50000:
        nn_train_convergence=2
    #nn_train_convergence=1 #early stopping for debugging 
##############################################################################
#Post training, plot results 
#Final unpacking of hyperparam vector 
knn_alpha=unpack.alpha(beta, knn_nn, knn_nb)
knn_beta1=unpack.beta1(beta, knn_nn, knn_nb,data_out_dim)
knn_beta0=unpack.beta2(beta,knn_nn, knn_nb,data_out_dim)
knn_rho=unpack.rho(beta, knn_nn, knn_nb, data_out_dim)
knn_theta=unpack.theta(beta, knn_nn, knn_nb, data_out_dim)
knn_v=unpack.v(beta,knn_nn, knn_nb, data_out_dim,data_in_dim)
    
    
#Evaluate KNN for every x co-ordinate    
#y_full=[0]*(100*n+1) #time steps
y_full=np.zeros((100*n+1,1))
x=np.zeros((1,data_in_dim))
for k2 in range(0,len(data_tfull)):
    Fc_train=(data_Fcfull[k2]/data_norm)+0.5
#    x=[0]*data_in_dim
    x=data_tfull[k2]/data_norm_t
    y_full[k2]=ffNN.ff(knn_nn,knn_nb,knn_v,knn_alpha,knn_theta,knn_beta1,knn_beta0,Fc_train,x,\
                    knn_rho,data_out_dim)
    y_full[k2]=(y_full[k2]-0.5)*data_norm

fig, ax = plt.subplots()
ax.plot(t, y_full, 'g',label='KNN') 
ax.plot(t,sc, 'r',label='Coarse data')
ax.scatter(te,fe_plot,marker='+',label='Expensive data')
ax.legend()
ax.set_xlabel('t')
ax.set_ylabel('f(t)') 


#compare MSE for no KNN vs KNN 
#MSE_noKNN=0.5*np.sum(np.multiply(data_Fefull-data_Fcfull,data_Fefull-data_Fcfull))
#MSE_KNN=0.5*np.sum(np.multiply(data_Fefull-y_full,data_Fefull-y_full))
"""     

    
    """