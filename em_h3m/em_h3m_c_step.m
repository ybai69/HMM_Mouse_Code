function h3m = em_h3m_c_step(h3m,O,mopt)
if isfield(mopt,'reg_cov')
    reg_cov = mopt.reg_cov;
else
    reg_cov = 0;
end

if ~isfield(mopt,'min_iter')
    mopt.min_iter = 0;
end

if ~isfield(mopt,'story')
    mopt.story = 0;
end

if mopt.story
    story = {};
    time  = [];
end

here_time  = tic;
if isfield(mopt,'start_time')
    a_time = mopt.start_time;
else
    a_time = here_time;
end

if mopt.story
    story{end+1} = h3m;
    time(end+1)  = toc(a_time);
end

num_iter = 0;

N = size(h3m.hmm{1}.A,1);
M = (h3m.hmm{1}.emit{1}.ncentres);
I = length(O);
K = h3m.K;

[T dim]    = size(O{1});
%sample_num = mopt.sample_num;
h3m.LogLs  = [];

inf_norm = 1;
if isfield(mopt,'inf_norm')
    switch mopt.inf_norm
        case 't'
            inf_norm = T;
    end
end

while 1

    %%%%%%%%%%%%%%%%%%%%
    %%%    E-step    %%%
    %%%%%%%%%%%%%%%%%%%%
    
 

    % compute soft assignments
    [llt z_hat] = h3m_c_inference(h3m,O,inf_norm); z_hat(isnan(z_hat)) = 0;ll = z_hat .* llt;
    % set to zero the -inf * 0 terms
    ll(intersect(find(isinf(llt)),find((z_hat ==0)))) = 0;
    new_LogLikelihood_1 = ones(1,I) * ll * ones(K,1);
    
    log_h= 0;
    for j = 1 : K
        for n = 1 : N
            beta_mu = h3m.hmm{j}.emit{n}.beta_mu;
            tau_mu = h3m.hmm{j}.emit{n}.tau_mu;
            beta_sigma = h3m.hmm{j}.emit{n}.beta_sigma;
            tau_sigma = h3m.hmm{j}.emit{n}.tau_sigma;
            %mu = h3m.hmm{j}.emit{n}.mu;
            %vars = h3m.hmm{j}.emit{n}.vars;
            mu = [];
            vars= [];
            for i = 1 : I
                mu = [mu,h3m.hmm{j}.emit{n}.mu_samples{i}];
                vars = [vars, h3m.hmm{j}.emit{n}.vars_samples{i}];  
            end
            sample_len = length(mu)./I;
            h_mu = normpdf(mu,beta_mu,sqrt(tau_mu));
            h_sigma = lognpdf(vars,beta_sigma,sqrt(tau_sigma));
            log_h = log_h+sum(log(h_mu))./sample_len+sum(log(h_sigma))./sample_len;  
        end
    end

    new_LogLikelihood_2=log_h;
    
    
    new_LogLikelihood =new_LogLikelihood_1+new_LogLikelihood_2
    old_LogLikelihood = h3m.LogL;
    h3m.LogL          = new_LogLikelihood;
    h3m.LogLs(end+1)  = new_LogLikelihood;
    h3m.Z             = z_hat;
    
   if isnan(new_LogLikelihood)
        break
    end

    
    % check whether to continue or to return
    
    
    stop = 0;
    if num_iter > 1
        changeLL = (new_LogLikelihood - old_LogLikelihood) / abs(old_LogLikelihood);
    else
        changeLL = inf;
    end
        
    switch mopt.termmode
        case 'L'
            if (abs(changeLL) < mopt.termvalue) 
                stop = 1;
            end
        case 'T'
            time_elapsed4now = toc(here_time);
            if (time_elapsed4now >= mopt.termvalue) 
                stop = 1;
            end
            
    end
    
    if (num_iter > mopt.max_iter)
        stop = 1;
    end

    if (stop) && (num_iter >= mopt.min_iter)
        break
    end
    
    num_iter = num_iter + 1;
    disp(num_iter)

    %%%%%%%%%%%%%%%%%%%%
    %%%    M-step    %%%
    %%%%%%%%%%%%%%%%%%%%

    % compute new parameters

    h3m_new = h3m;

    % new weights (omega)
    omega_new = (ones(1,I) / I) * z_hat;
    h3m_new.omega = omega_new;


    % loop all the hmm components of the new mixture
    for j = 1 : K
        new_prior = zeros(N,1);
        new_A     = zeros(N,N);
        new_mix   = h3m.hmm{j}.emit;
        mu_out    = cell(N,I);
        var_out   = cell(N,I);
        outer     = cell(1,N);
        % put to zero all fields
        for n = 1 : N
            new_mix{n}.priors  = new_mix{n}.priors  *0;
            new_mix{n}.beta_mu = new_mix{n}.beta_mu *0;
            outer{n}           = new_mix{n}.beta_mu  *0;
        end
        
        for i = 1 : I
            [foo gamma_start epsilon_sum gamma_eta_sum gamma_eta_y_sum gamma_eta_y2_sum] ...
                =  get_ss_hmm_c_scaled(h3m.hmm{j},O{i},i);
            
            
            gamma_start(isnan(gamma_start))           = 0;
            epsilon_sum(isnan(epsilon_sum))           = 0;
            gamma_eta_sum(isnan(gamma_eta_sum))       = 0;
            gamma_eta_y_sum(isnan(gamma_eta_y_sum))   = 0;
            gamma_eta_y2_sum(isnan(gamma_eta_y2_sum)) = 0;
            
            if z_hat(i,j) > 0
                new_prior = new_prior + z_hat(i,j) * gamma_start;
                new_A     = new_A     + z_hat(i,j) * epsilon_sum;
                
                
                npar = 2*N;             % number of unknowns
                    
                obs     = O{i};
                A       = h3m.hmm{j}.A;
                prior   = h3m.hmm{j}.prior;
                    
                beta_mu    = []; tau_mu     = [];
                beta_sigma = []; tau_sigma  = [];
                for ii = 1 : N
                    beta_mu    = [beta_mu,h3m.hmm{1}.emit{ii}.beta_mu];
                    tau_mu     = [tau_mu,h3m.hmm{1}.emit{ii}.tau_mu];
                    beta_sigma = [beta_sigma,h3m.hmm{1}.emit{ii}.beta_sigma];
                    tau_sigma  = [tau_sigma,h3m.hmm{1}.emit{ii}.tau_sigma ];
                end
                
                data = struct('A',A,'prior',prior,'obs',obs,'beta_mu',beta_mu,'tau_mu',tau_mu,'beta_sigma',beta_sigma,'tau_sigma',tau_sigma);
                for ii = 1 : npar/2
                    params{ii} = {sprintf('mu_%d',ii),h3m.hmm{j}.emit{ii}.beta_mu};
                end
                for ii = (npar/2+1) : npar
                    params{ii} = {sprintf('sigma_%d',ii-npar/2),exp(h3m.hmm{j}.emit{ii-npar/2}.beta_sigma),0,Inf};
                end
                    
                model.ssfun     = @hmmss;
                    
                options.method  = 'dram';
                options.nsimu   = mopt.nsimu_dram;
                %options.nsimu   = 5000;
                %options.nsimu   = 1000;
                %options.nsimu   = 2000;
                options.qcov    = eye(npar); % [initial] proposal covariaance
                %options.qcov    = h3m.hmm{j}.qcov{i}; % [initial] proposal covariaance
                [results,chain] = mcmcrun(model,data,params,options); 
                    
                 options.method  = 'mh';
                 options.nsimu   = mopt.nsimu_mh;
                 %options.nsimu   = 2000;
                 options.qcov    = results.qcov; % [initial] proposal covariaance
                 for ii = 1 : npar/2
                     params{ii}  = {sprintf('mu_%d',ii),mean(chain(end/2:end,ii))};
                 end
                 for ii = (npar/2+1) : npar
                     params{ii}  = {sprintf('sigma_%d',ii-npar/2),mean(chain(end/2:end,ii)),0,Inf};
                 end
                 [results,chain] = mcmcrun(model,data,params,options);
                 
                 h3m_new.hmm{j}.qcov{i}  = results.qcov  

                for n = 1 : N
                    new_mix{n}.priors = new_mix{n}.priors  + z_hat(i,j) * gamma_eta_sum(n,:);
                    mu_out{n,i}= chain(end/2:end,n);
                    var_out{n,i}=chain(end/2:end,n + N);
                end
                
            else
                pippo = 0;
            end
            
        end

        % normalize things
        new_prior = new_prior / sum(new_prior);
        new_A     = new_A    ./ repmat(sum(new_A,2),1,N);
        
        
        for n = 1 : N
            mu_ln     = [];
            var_ln    = [];
            mu_result =[];
            var_result=[];
            for i =1 : I
                mu_ln = [mu_ln;mu_out{n,i}];
                var_ln = [var_ln;var_out{n,i}];
                mu_result=[mu_result mean(mu_out{n,i})];
                var_result= [var_result mean(var_out{n,i})];
                new_mix{n}.mu_samples{i} = mu_out{n,i};
                new_mix{n}.vars_samples{i} = var_out{n,i};
                
            end
            new_mix{n}.beta_mu = mean(mu_ln);
            new_mix{n}.tau_mu = var(mu_ln);
            new_mix{n}.beta_sigma = mean(log(var_ln));
            new_mix{n}.tau_sigma = var(log(var_ln));
            new_mix{n}.mu = mu_result;
            new_mix{n}.vars = var_result;
            new_mix{n}.covars = exp(new_mix{n}.beta_sigma+new_mix{n}.tau_sigma/2)+new_mix{n}.tau_mu;
            new_mix{n}.priors  = new_mix{n}.priors  ./ sum(new_mix{n}.priors);
        end

        h3m_new.hmm{j}.prior = new_prior;
        h3m_new.hmm{j}.A     = new_A;
        h3m_new.hmm{j}.emit  = new_mix;
        
        
    end

    
    h3m = h3m_new;

    if mopt.story
        story{end+1} = h3m;
        time(end+1) = toc(a_time);
    end
    
end

h3m.elapsed_time = toc(here_time);

end

