#!/usr/local/bin/python3
import numpy as np
if not __file__.endswith('_hmm_gaussian.py'):
    print('ERROR: This file is not named correctly! Please name it as Lastname_hmm_gaussian.py (replacing Lastname with your last name)!')
    exit(1)

DATA_PATH = "/Users/jonathanyakubov/Desktop/MachineLearning/MLhw8/em/" #TODO: if doing development somewhere other than the cycle server (not recommended), then change this to the directory where your data file is (points.dat)

def parse_data(args):
    num = float
    dtype = np.float32
    data = []
    with open(args.data_file, 'r') as f:
        for line in f:
            data.append([num(t) for t in line.split()])
    dev_cutoff = int(.9*len(data))
    train_xs = np.asarray(data[:dev_cutoff],dtype=dtype)
    dev_xs = np.asarray(data[dev_cutoff:],dtype=dtype) if not args.nodev else None
    return train_xs, dev_xs

def init_model(args):
    if args.cluster_num:
        mus = np.zeros((args.cluster_num,2))
        for i in range(args.cluster_num):
        	for n in range(2):
        		mus[i][n]=np.random.normal(loc=1,scale=2)
        if not args.tied:
            sigmas = np.zeros((args.cluster_num,2,2))
            for i in range(len(sigmas)):
            	sigmas[i]=np.identity(2)
            	rand_integer=np.random.randint(1,10)
            	sigmas[i][0][0]=rand_integer
            	sigmas[i][1][1]=rand_integer
        else:
        	sigmas = np.zeros((2,2))
        	sigmas=np.identity(2)
        	rand_integer=np.random.randint(2,5)
        	sigmas[0][0]=rand_integer
        	sigmas[1][1]=rand_integer
        transitions = np.zeros((args.cluster_num,args.cluster_num)) #transitions[i][j] = probability of moving from cluster i to cluster j
        for i in range(len(transitions)):
        	for j in range(len(transitions)):
        		transitions[i][j]=1/int(len(transitions))    	
        #transitions are the hidden sequences... #of clusters=# hidden states 
        initials = np.zeros(args.cluster_num) #probability for starting in each state
        for i in range(len(initials)):       #initialization initials
        	initials[i]=1/int(len(initials))
        #TODO: randomly initialize clusters (mus, sigmas, initials, and transitions)
       #  raise NotImplementedError #remove when random initialization is implemented
    else:
        mus = []
        sigmas = []
        transitions = []
        initials = []
        with open(args.clusters_file,'r') as f:
            for line in f:
                #each line is a cluster, and looks like this:
                #initial mu_1 mu_2 sigma_0_0 sigma_0_1 sigma_1_0 sigma_1_1 transition_this_to_0 transition_this_to_1 ... transition_this_to_K-1
                vals = list(map(float,line.split()))
                initials.append(vals[0])
                mus.append(vals[1:3])
                sigmas.append([vals[3:5],vals[5:7]])
                transitions.append(vals[7:])
        initials = np.asarray(initials)
        transitions = np.asarray(transitions)
        mus = np.asarray(mus)
        sigmas = np.asarray(sigmas)
        args.cluster_num = len(initials)

    #TODO: Do whatever you want to pack mus, sigmas, initals, and transitions into the model variable (just a tuple, or a class, etc.)
    model = (mus,sigmas,initials,transitions)
    # raise NotImplementedError #remove when model initialization is implemented
    return model

def forward(model, data, args):
	mus,sigmas,initials,transitions=model
	from scipy.stats import multivariate_normal
	from math import log
	alphas = np.zeros((len(data),args.cluster_num))
	omissions=np.zeros((len(data),args.cluster_num))
	log_likelihood = 0.0
	for t in range(len(data)):   #observation at time t
		for j in range(args.cluster_num):   #hidden state
			if t==0:  #if the observation is the first 
				if args.tied:
					alphas[t][j]=1*initials[j]*multivariate_normal.pdf(data[t],mean=mus[j],cov=sigmas)
					omissions[t][j]=multivariate_normal.pdf(data[t],mean=mus[j],cov=sigmas)
				else:	
					alphas[t][j]=1*initials[j]*multivariate_normal.pdf(data[t],mean=mus[j],cov=sigmas[j])
					omissions[t][j]=multivariate_normal.pdf(data[t],mean=mus[j],cov=sigmas[j])
			else:
				if args.tied:
					alphas[t][j]=np.matmul(alphas[t-1,:],transitions[:,j]*multivariate_normal.pdf(data[t],mean=mus[j],cov=sigmas))
					omissions[t][j]=multivariate_normal.pdf(data[t],mean=mus[j],cov=sigmas)		
				else:
					alphas[t][j]=np.matmul(alphas[t-1,:],transitions[:,j]*multivariate_normal.pdf(data[t],mean=mus[j],cov=sigmas[j]))
					omissions[t][j]=multivariate_normal.pdf(data[t],mean=mus[j],cov=sigmas[j])
	
		normalization=np.sum(alphas[t,:])
		alphas[t,:]=alphas[t,:]/normalization
		log_likelihood+=log(normalization)
	# print(alphas)
	# print(alphas,log_likelihood, omissions)
	return alphas, log_likelihood, omissions
	
		
    	
    #TODO: Calculate and return forward probabilities (normalized at each timestep; see next line) and log_likelihood
    #NOTE: To avoid numerical problems, calculate the sum of alpha[t] at each step, normalize alpha[t] by that value, and increment log_likelihood by the log of the value you normalized by. This will prevent the probabilities from going to 0, and the scaling will be cancelled out in train_model when you normalize (you don't need to do anything different than what's in the notes). This was discussed in class on April 3rd.
    # raise NotImplementedError

def backward(model, data, args):
	mus,sigmas,initials,transitions=model
	alphas,log_likelihood, omissions=forward(model, data, args)
	from scipy.stats import multivariate_normal
	betas = np.zeros((len(data),args.cluster_num))
	emissions=np.zeros((len(data),args.cluster_num))
	for t in range(len(data)-1,-1,-1):
		for j in range(args.cluster_num):
			if t==len(data)-1:
				if args.tied:
					betas[t][j]=1
					emissions[t][j]=multivariate_normal.pdf(data[t],mean=mus[j],cov=sigmas)
				else:	
					betas[t][j]=1
					emissions[t][j]=multivariate_normal.pdf(data[t],mean=mus[j],cov=sigmas[j])
			else:
				if args.tied:
					betas[t][j]=np.sum(betas[t+1,:]*transitions[j,:]*omissions[t+1,:])
					# betas[t][j]=np.matmul(betas[t+1,:],transitions[j,:]*multivariate_normal.pdf(data[t+1],mean=mus[j],cov=sigmas))
					emissions[t][j]=multivariate_normal.pdf(data[t],mean=mus[j],cov=sigmas)
				else:
					# betas[t][j]=np.matmul(betas[t+1,:],transitions[j,:]*multivariate_normal.pdf(data[t+1],mean=mus[j],cov=sigmas[j]))
					betas[t][j]=np.sum(betas[t+1,:]*transitions[j,:]*omissions[t+1,:])
					emissions[t][j]=multivariate_normal.pdf(data[t],mean=mus[j],cov=sigmas[j])
		normalization=np.sum(betas[t,:])
		betas[t,:]=betas[t,:]/normalization
	# print(betas)
	# print(emissions)
# 	print(emissions.shape)
	return betas
    
    #TODO: Calculate and return backward probabilities (normalized like in forward before)
    # raise NotImplementedError
    

def train_model(model, train_xs, dev_xs, args):
    from scipy.stats import multivariate_normal
    # from matplotlib import pyplot as plt
    mus,sigmas,initials,transitions=model
   #  dev_likelihoods=[] #initialization of dev likelihoods
#     train_likelihoods=[] #initialization of train  likelihoods
#     iterations=[]   #initialization of iterations
    for itr in range(args.iterations):
    	# iterations.append(itr+1)
    	# print(p+1)
    	alphas,ll,omissions=forward(model,train_xs,args)  #Expectation step 
   #  print(alphas)
#     print(omissions)
#     print(omissions.shape)
	
    	betas=backward(model,train_xs,args)
    # print(betas)
    	gamma=np.zeros((len(train_xs),args.cluster_num))
    	gamma_t_k_array=np.zeros((len(train_xs),args.cluster_num),dtype=object)
    	ksi=np.zeros((len(train_xs),args.cluster_num,args.cluster_num),dtype=object)  #i might need to subtract 1 
    	sigmas_array=np.zeros((len(train_xs),args.cluster_num),dtype=object)
    	for t in range(len(train_xs)):
    		for j in range(args.cluster_num):
    			gamma[t][j]=alphas[t][j]*betas[t][j]/(np.sum(alphas[t,:]*betas[t,:]))
    			for k in range(args.cluster_num):  #i might need to drop one here, maybe the top, depends on the initialization 
    				if args.tied:
    					ksi[t][j][k]=alphas[t-1][j]*betas[t][k]*transitions[j,k]*multivariate_normal.pdf(train_xs[t],mean=mus[k],cov=sigmas)
    				else:
    					ksi[t][j][k]=alphas[t-1][j]*betas[t][k]*transitions[j,k]*multivariate_normal.pdf(train_xs[t],mean=mus[k],cov=sigmas[k])
    		ksi[t,:,:]=ksi[t,:,:]/np.sum(ksi[t,:,:]) 
    # print(ksi)   		
     
    
    
    	for k in range(args.cluster_num):  #M-step  #mus calculation and pie calculations 
    		initials[k]=gamma[0][k]  #pie calculation 
    	
    		for i in range(len(train_xs)):	  # mus calculations 
    				gamma_t_k_array[i][k]=gamma[i][k]*train_xs[i] #scalar times datapoint with 2D's
    		column_gamma_t_k=gamma_t_k_array[:,k]
    		mus[k]=(np.sum(column_gamma_t_k))/np.sum(gamma[:,k])
    	# print(mus[k])  this is fine
    	
    	
    		if not args.tied:
    			for n in range(len(train_xs)):   #sigmas calculation #only for tied case 
    				x=np.reshape(train_xs[n], (2,1))
    				mu=np.reshape(mus[k],(2,1))
    				x_n_mu_difference=x-mu
    				product=np.matmul(x_n_mu_difference,x_n_mu_difference.T)
    				sigmas_array[n][k]=gamma[n][k]*product  #sigmas array 
    			column_k_sigma_x_n=sigmas_array[:,k]
    			sigmas[k]=(np.sum(column_k_sigma_x_n))/(np.sum(gamma[:,k])) 
    		# print(sigmas[k]) this is fine
    # print(gamma)	this is fine	
    	# print(sigmas)
    	if args.tied:   #sigmas calculation for tied case 
    		for j in range(len(train_xs)):
    				for d in range(args.cluster_num):
    					x=np.reshape(train_xs[j], (2,1))
    					mu=np.reshape(mus[d],(2,1))
    					x_n_mu_difference=x-mu
    					sigmas+=gamma[j][d]*np.matmul(x_n_mu_difference,x_n_mu_difference.T)
    		sigmas=(sigmas/len(train_xs)) 
    		# print(sigmas)		
    	
    	for k in range(args.cluster_num):  #transitions calculations
    		for j in range(args.cluster_num):
    		# print((np.sum(ksi[:,j,k])))
#     		print((np.sum(gamma[:,k],axis=0)))
    			transitions[k][j]=(np.sum(ksi[1:,k,j]))/(np.sum(gamma[:,k],axis=0))
    		# p# rint(transitions)
    	model=(mus,sigmas,initials, transitions)
  #   	if not args.nodev:
#     		dev_likelihood=average_log_likelihood(model, dev_xs,args)
#     		dev_likelihoods.append(dev_likelihood)
#     		train_likelihood=average_log_likelihood(model, train_xs,args)
#     		train_likelihoods.append(train_likelihood)
#     if not args.nodev:
#     	plt.plot(iterations,train_likelihoods, label="Training Data Avg Log Likelihood")
#     	plt.plot(iterations,dev_likelihoods, label="Dev Data Avg Log Likelihood")
#     	plt.xlabel("Iterations")
#     	plt.ylabel("Average Log Likelihood")
#     	plt.title("Average Log Likelihood vs. Iterations")
#     	plt.legend()
#     	plt.show()	
   
   
   
   
    return model
			
    

def average_log_likelihood(model, data, args):
	mus,sigmas,initials,transitions=model
	alphas, log_likelihood,omissions=forward(model,data,args)
	ll=log_likelihood/len(data)
	return ll
    

def extract_parameters(model):
    #TODO: Extract initials, transitions, mus, and sigmas from the model and return them (same type and shape as in init_model)
    initials = model[2]
    transitions = model[3]
    mus = model[0]
    sigmas = model[1]
    # raise NotImplementedError #remove when parameter extraction is implemented
    return initials, transitions, mus, sigmas

def main():
    import argparse
    import os
    print('Gaussian') #Do not change, and do not print anything before this.
    parser = argparse.ArgumentParser(description='Use EM to fit a set of points')
    init_group = parser.add_mutually_exclusive_group(required=True)
    init_group.add_argument('--cluster_num', type=int, help='Randomly initialize this many clusters.')
    init_group.add_argument('--clusters_file', type=str, help='Initialize clusters from this file.')
    parser.add_argument('--nodev', action='store_true', help='If provided, no dev data will be used.')
    parser.add_argument('--data_file', type=str, default=os.path.join(DATA_PATH, 'points.dat'), help='Data file.')
    parser.add_argument('--print_params', action='store_true', help='If provided, learned parameters will also be printed.')
    parser.add_argument('--iterations', type=int, default=1, help='Number of EM iterations to perform')
    parser.add_argument('--tied',action='store_true',help='If provided, use a single covariance matrix for all clusters.')
    args = parser.parse_args()
    if args.tied and args.clusters_file:
        print('You don\'t have to (and should not) implement tied covariances when initializing from a file. Don\'t provide --tied and --clusters_file together.')
        exit(1)

    train_xs, dev_xs = parse_data(args)
    model = init_model(args)
    model = train_model(model, train_xs, dev_xs, args)
    nll_train = average_log_likelihood(model, train_xs, args)
    print('Train LL: {}'.format(nll_train))
    if not args.nodev:
        nll_dev = average_log_likelihood(model, dev_xs, args)
        print('Dev LL: {}'.format(nll_dev))
    initials, transitions, mus, sigmas = extract_parameters(model)
    if args.print_params:
        def intersperse(s):
            return lambda a: s.join(map(str,a))
        print('Initials: {}'.format(intersperse(' | ')(np.nditer(initials))))
        print('Transitions: {}'.format(intersperse(' | ')(map(intersperse(' '),transitions))))
        print('Mus: {}'.format(intersperse(' | ')(map(intersperse(' '),mus))))
        if args.tied:
            print('Sigma: {}'.format(intersperse(' ')(np.nditer(sigmas))))
        else:
            print('Sigmas: {}'.format(intersperse(' | ')(map(intersperse(' '),map(lambda s: np.nditer(s),sigmas)))))

if __name__ == '__main__':
    main()