function varargout = gmmprob_random(varargin)
%GMMPROB Computes the data probability for a Gaussian mixture model.
% [prob prob_comp] = gmmprob(mix, x)
%	Description
%	 This function computes the unconditional data density P(X) for a
%	Gaussian mixture model.  The data structure MIX defines the mixture
%	model, while the matrix X contains the data vectors.  Each row of X
%	represents a single vector.
%
%	See also
%	GMM, GMMPOST, GMMACTIV
%

%	Copyright (c) Ian T Nabney (1996-2001)
% modified by Cmanuel Coviello (March, 2011)

% Check that inputs are consistent
% errstring = consist(mix, 'gmm', x);
% if ~isempty(errstring)
%   error(errstring);
% end

% Compute activations
if nargin == 3
    mix = varargin{1};
    x = varargin{2};
    num = varargin{3};
    ndata = size(x, 1);
    a = zeros(ndata, mix.ncentres);  % Preallocate matrix
    for j = 1:mix.ncentres
        a(:, j) =  normpdf(x,mix.mu(num),sqrt(mix.vars(num))); 
    end
    prob = a * (mix.priors)';


    varargout{1} = prob;
    prob_comp = a .* (ones(size(a,1),1) * mix.priors);
% it is T by M 
%(normalize)
    prob_comp = prob_comp ./ (sum(prob_comp,2) * ones(1,size(prob_comp,2)));
    varargout{2} = prob_comp;



elseif nargin == 4
    mix = varargin{1};
    x = varargin{2};
    num = varargin{3};
    sample_num=varargin{4};
    ndata = size(x, 1);
    a = zeros(ndata, mix.ncentres);  % Preallocate matrix
    for j = 1:mix.ncentres
        a(:, j) =  normpdf(x,mix.mu_samples{num}(sample_num),sqrt(mix.vars_samples{num}(sample_num))); 
    end
    prob = a * (mix.priors)';


    varargout{1} = prob;
    prob_comp = a .* (ones(size(a,1),1) * mix.priors);
% it is T by M 
%(normalize)
    prob_comp = prob_comp ./ (sum(prob_comp,2) * ones(1,size(prob_comp,2)));
    varargout{2} = prob_comp;


else
    error('Input sould be an emission\n')
end



% for i= 1 : ndata
%     for j = 1:mix.ncentres
%         beta_mu = mix.beta_mu(j); 
%         beta_sigma= mix.beta_sigma(j);  
%         tau_mu = mix.tau_mu(j); 
%         tau_sigma = mix.tau_sigma(j); 
%         gmm_random_pdf = @(z, y) 1./sqrt(2*pi*(z+tau_mu)).*exp(-1./(2.*(z+tau_mu)).*(beta_mu-y).^2).*1./(z.*sqrt(2*pi*tau_sigma)).*exp(-1./(2*tau_sigma).*(log(z)-beta_sigma).^2);   
%         a(i, j) = integral(@(z) gmm_random_pdf(z ,x(i)),0,Inf);
%     end
% end






% faster
% % prob_comp = a .* (ones(size(a,1),1) * mix.priors);
% % % it is T by M 
% % % (normalize)
% % prob_comp = prob_comp ./ (sum(prob_comp,2) * ones(1,size(prob_comp,2)));


% move to log-scale
% a = gmmactiv_bis(mix, x);
% prob_comp = a + (ones(size(a,1),1) * log(mix.priors));
% prob_comp = exp(prob_comp-logtrick(prob_comp')'*ones(1,size(prob_comp,2)));
% 

