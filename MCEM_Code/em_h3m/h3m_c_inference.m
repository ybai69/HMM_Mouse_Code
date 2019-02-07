function [llt z_hat] = h3m_c_inference(h3m,O,normalization)

% O = {obs1 obs2 obs3 ... obsI}
% h3m.K
% h3m.hmm = {hmm1 hmm2 ... hmmK}
% h3m.omega = [omega1 omega2 ... omegaK]
% hmm1.A hmm1.B hmm1.prior

if ~exist('normalization','var') || isempty(normalization)
    normalization = 1;
end


N = size(h3m.hmm{1}.A,1);
M = (h3m.hmm{1}.emit{1}.ncentres);
[T, dim] = size(O{1});






    %%%%%%%%%%%%%%%%%%%%
    %%%    inference    %%%
    %%%%%%%%%%%%%%%%%%%%

    % compute soft assignments

    I = length(O);
    K = h3m.K;
    
    
    

    LL = zeros(I,K);

    for i = 1 : I
        for j = 1 : K
             [foo foo2(i,j) foo3 ] = fwd_hmm_c_scaled(h3m.hmm{j}.prior,h3m.hmm{j}.A,h3m.hmm{j}.emit,O{i},i);
             LL(i,j) = sum(log(foo3));
      
        end
    end
    
    
    LL = LL/normalization;
    
    if any(LL(:) == -Inf)
        fprintf('\nThere are -inf terms in the Log likelihoods')
    end
    
    if any(isnan(LL(:)))
        fprintf('\nThere are NaN terms in the Log likelihoods!!!!!!!!')
    end
 
    % add prior
    LL_wP  = LL + ones(I,1) * log(h3m.omega);
    LL_wPm = logtrick(LL_wP')';
    
    LL_wPz = LL_wP - LL_wPm * ones(1,K);
    
    z_hat = exp(LL_wPz);
    % normalize (in case it is not)
    z_hat = z_hat ./ (z_hat * ones(K,K));
    
   llt =  LL_wP;
   % llt= LL
    
    
end