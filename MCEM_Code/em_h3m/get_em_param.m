function [mopt, emopt] = get_em_param(N,M,K,init,termmode,termvalue,...
    max_iter,min_iter,max_obs,reg_cov,trials,initial_val,seed,nsimu_dram,nsimu_mh)

mopt.N = N;
mopt.M = M;
mopt.emit.type= 'gmm';
mopt.emit.covar_type= 'diag';
mopt.K = K;

mopt.initial_val = initial_val;
mopt.seed = seed;
%mopt.sample_num = sample_num;


mopt.initmode = init; 
% rule for termination of EM iterations
mopt.termmode = termmode;
mopt.termvalue = termvalue;
% max number of iterations
mopt.max_iter = max_iter;
% max number of time series to use for learning
mopt.max_obs = max_obs;
mopt.min_iter = min_iter;

emopt.trials = trials;


mopt.reg_cov = reg_cov;
mopt.nsimu_dram = nsimu_dram;
mopt.nsimu_mh = nsimu_mh;

% mopt.initmode =
% 'r'    random initialization of all parameters (this will probably never work well for you)
% 'p'    randomly partition the input time series in K group, and estimate each HMM component on one of the partition
% 'g'    first estimate a GMM on all the data, than initialize each HMM by setting emission to the GMM (with randomized weights) and using random parameters for the HMM dynamics
% 'gm'   similar to 'g', but uses MATLAB own function 
% 'km'   similar to 'g' but uses k means
% 'gL2R' similar to 'g'. but initialize HMMs as left to righ HMMs
% 'c' customized initialization 



