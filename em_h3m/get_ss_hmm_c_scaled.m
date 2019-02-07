function [ll gamma_start epsilon_sum ...
         gamma_eta_sum gamma_eta_y_sum gamma_eta_y2_sum ] =  get_ss_hmm_c_scaled(varargin)

% [] =  get_ss_hmm(prior,A,B,obs)
% 
% gets sufficient statistics
% 
% input:
% prior is a N x 1 vector
% A is a N x N transition matrix
% B is a N x T observation matrix
% obs is a series of T observations
% 
% output:
% ll is the log-likelihood log P(obs)
% gamma_start is a N x 1 vector: expected starts in 1;
% epsilon_sum is a N x N matrix: expected number of trasients from u to v
% gamma_sum   is a N x M matrix: expected number of time u outputs k
%
% coded by Emanuele Coviello (11/24/2010)

if nargin == 3
    prior = varargin{1}.prior;
    A     = varargin{1}.A;
    emit     = varargin{1}.emit;
    obs     = varargin{2};
    num     = varargin{3};
elseif nargin == 5
    prior = varargin{1};
    A     = varargin{2};
    emit     = varargin{3};
    obs     = varargin{4};
    num     = varargin{5};
else
    error('Input sould be an hmm\n')
end

[T dim] = size(obs);
[N] = size(A,1);
M = emit{1}.ncentres;

gamma_start=zeros(N,1);
epsilon_sum=zeros(N,N);
gamma_eta_sum = zeros(N,M);
gamma_eta_y_sum = zeros(N,dim,M);
gamma_eta_y2_sum = zeros(N,dim,M);
sample_num = length(emit{1}.mu_samples);
for k = 1 : sample_num
    
B = zeros(N,T);
eta = zeros(N,T,M);
for n = 1 : N
    [probs eta(n,:,:)] = gmmprob_random(emit{n}, obs,num,k);
    probs(isnan(probs)) = 0;
    B(n,:) = probs(:)'; % make it a row
end

[alpha ll scale] =  fwd_hmm_c_scaled(prior,A,B);
beta =  bck_hmm_c_scaled(A,B,scale);


% compute epsilon

epsilon = zeros(N,N,T-1);

for t = 1 : T-1
    epsilonFoo = ((alpha(:,t) * beta(:,t+1)') .* A) * diag(B(:,t+1));
    epsilon(:,:,t) = epsilonFoo / scale(t+1);
end

% compute gamma

gamma = zeros(N,T);

for t = 1 : T
    gamma(:,t) = alpha(:,t).*beta(:,t);
end
% gamma_(N,T) -- probability p(z_t=n|obs)
% compute the sufficient stats:

% to estimate initial state probabilities
gamma_start = gamma_start+gamma(:,1);

% to estimate transition probabilities
% epsilon_sum is a N x N matrix: expected number of trasients from u to v
epsilon_sum = epsilon_sum+ sum(epsilon,3);
% compute nu
for m = 1 : M
    eta_m = squeeze(eta(:,:,m)); 
    gamma_eta_sum(:,m) = gamma_eta_sum(:,m)+sum(gamma .* eta_m,2)./sample_num; 
    % N by 1
    gamma_eta_y_sum(:,:,m) =gamma_eta_y_sum(:,:,m)+ ((gamma .* eta_m) * obs)./sample_num;
    %gamma_eta_y(:,:,m) = ((gamma .* eta_m)' .* obs)';
    gamma_eta_y2_sum(:,:,m) =gamma_eta_y2_sum(:,:,m)+ ((gamma .* eta_m) * (obs.^2))./sample_num;     
end
    
end

%normalize
gamma_start= gamma_start./sample_num;
epsilon_sum = epsilon_sum ./sample_num;




