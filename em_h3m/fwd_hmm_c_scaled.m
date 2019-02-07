function [alpha ll_log scale ] =  fwd_hmm_c_scaled(varargin)

% [alpha l] =  fwd_hmm(prior,A,B,obs)
% 
% does forward algorithm
% 
% input:
% prior is a N x 1 vector
% A is a N x N transition matrix
% B is a N x T observation matrix
% obs is a series of T observations
% 
% output:
% alpha is a N x T matrix with alpha(u,t) = P(obs(1:t),q_t = u)
% l is the likelihood P(obs(1:T))
%
% coded by Emanuele Coviello (11/24/2010)

if nargin == 3
    prior = varargin{1};
    A     = varargin{2};
    B     = varargin{3};
    T = size(B,2);
    N = size(A,1);

elseif nargin == 5
    prior = varargin{1};
    A     = varargin{2};
    emit     = varargin{3};
    obs     = varargin{4};
    num=varargin{5};


    % create probability of each observation in all the states
    M = emit{1}.ncentres;
    N = size(A,1);
    T = size(obs,1);
    B = zeros(N,T);
    for n = 1 : N
        probs = gmmprob_random(emit{n}, obs,num);
        probs(isnan(probs)) = 0;
        B(n,:) = probs(:)'; % make it a row
    end

else
    error('Input sould be an hmm\n')
end


delta = zeros(N,T);
alpha = zeros(N,T);
% initialilze

delta(:,1) = prior .* B(:,1);

% scale, otherwise the probability will be too small at the end
foo = sum(delta(:,1));
alpha(:,1) = delta(:,1) / foo;
scale(1) = foo;

% loop

for t = 1 : T - 1
    delta(:,t+1) = (A' * alpha(:,t)).* B(:,t+1);
    foo = sum(delta(:,t+1));
    alpha(:,t+1) = delta(:,t+1) / foo;
    scale(t+1) = foo;
end

ll_log = log(scale(end));

