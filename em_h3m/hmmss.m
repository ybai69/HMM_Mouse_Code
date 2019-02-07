function ss = hmmss(theta,data)
obs          = data.obs;
A            = data.A;
prior        = data.prior;
beta_mu      = data.beta_mu;
tau_mu       = data.tau_mu;
beta_sigma   = data.beta_sigma;
tau_sigma    = data.tau_sigma;
emit{1}.mu   = theta(1);
emit{2}.mu   = theta(2);
emit{3}.mu   = theta(3);
emit{1}.vars = theta(4);
emit{2}.vars = theta(5);
emit{3}.vars = theta(6);

M = 1;
N = size(A,1);
T = size(obs,1);
B = zeros(N,T);
for n = 1 : N
    probs  = normpdf(obs,emit{n}.mu,sqrt(emit{n}.vars));
    probs(isnan(probs)) = 0;
    B(n,:) = probs(:)'; % make it a row
end

delta = zeros(N,T);
alpha = zeros(N,T);

% initialilze
delta(:,1) = prior .* B(:,1);
scale(1)   = sum(delta(:,1));
alpha(:,1) = delta(:,1) / scale(1);

% loop
for t = 1 : T - 1
    delta(:,t+1) = (A' * alpha(:,t)).* B(:,t+1);
    scale(t+1)   = sum(delta(:,t+1));
    alpha(:,t+1) = delta(:,t+1) / scale(t+1);  
end

ll_1 = sum(log(scale));
ll_2 = 0;
for i = 1 : N
    %log(normpdf(emit{i}.mu,beta_mu(i),sqrt(tau_mu(i))))
    %log(lognpdf(emit{i}.vars,beta_sigma(i),sqrt(tau_sigma(i))))
    ll_2 = ll_2 -0.5 * ((emit{i}.mu - beta_mu(i))./sqrt(tau_mu(i))).^2-0.5*log(2*pi)-0.5*log(tau_mu(i));
    ll_2 = ll_2 -0.5 * ((log(emit{i}.vars) - beta_sigma(i))./sqrt(tau_sigma(i))).^2-0.5*log(2*pi)-0.5*log(tau_sigma(i))-log(emit{i}.vars);
end

ss   = -2*ll_1-2*ll_2;

