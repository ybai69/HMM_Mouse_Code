function [beta] =  bck_hmm_c_scaled(A,emit,scale)

% [beta] =  bck_hmm(A,B,obs)
% 
% does backward algorithm
% 
% input:
% A is a N x N transition matrix
% N states
% B is a N x T observation matrix
% M components of gmm
% obs is a series of T observations
% 
% output:
% beta is a N x T matrix with beta(u,t) = P(obs(t+1:T)|q_t = u)
%
% coded by Emanuele Coviello (11/24/2010)

% if ~exist('obs','var')
    B = emit;
    T = size(B,2);
    log_B = log(B);
    N = size(A,1);
% else
%     % create probability of each observation in all the states
%     T = size(obs,1);
%     M = emit{1}.ncentres;
%     N = size(A,1);
%     B = zeros(N,T);
%     log_B = zeros(N,T);
%     for n = 1 : N
%         probs = gmmprob_random(emit{n}, obs,num);
%         B(n,:) = probs(:)'; % make it a row
%         %probs = gmmprob_random_bis(emit{n}, obs);
%         %log_B(n,:) = probs(:)';
%         %log_B(n,:)=log(B);
%     end
% end

delta = zeros(N,T);
%log_beta = zeros(N,T);
beta = zeros(N,T);
% initialilze

%delta(:,T) = ones(N,1);
%log_beta(:,T) = zeros(N,1);
%foo = sum(delta(:,T));
beta(:,T) = ones(N,1);

% loop

for t = T - 1 : -1 : 1
    delta(:,t) = A * ( B(:,t+1) .* beta(:,t+1));
    %log_beta(:,t) = logtrick(log(A') + ( log_B(:,t+1) + beta(:,t+1)) * ones(1,N))';
    %foo = sum(beta(:,t));
    beta(:,t) = delta(:,t)/scale(t+1);
end

%log_beta= log(beta);

% T = length(obs);
% [N M] = size(B);
% 
% beta = zeros(N,T);
% 
% % initialilze
% 
% beta(:,T) = ones(N,1);
% 
% foo = sum(beta(:,T));
% beta(:,T) = beta(:,T)/foo;
% 
% % loop
% 
% for t = T - 1 : -1 : 1
% %     beta(:,t) = (A * B(:,obs(t+1))) .* beta(:,t+1);
%     beta(:,t) = A * ( B(:,obs(t+1)) .* beta(:,t+1));
%     foo = sum(beta(:,t));
%     beta(:,t) = beta(:,t)/foo;
%     
% end
% 
