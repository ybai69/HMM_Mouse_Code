function h3m_out = em_h3m_c(observations,mopt,emopt)


% need some functions from the gmm package... 


LL_best = -inf;

re_do = 0;

if isfield(emopt,'trials')
    
    % do multiple trials and keep the best
    
    while (LL_best == -inf) || (isnan(LL_best))
    
        for t = 1 : emopt.trials

            %addpath('gmm')

            % initialize h3m
            
            % number of input time series
            I = length(observations);
            K = mopt.K;
            N = mopt.N;
            
            % if there are too many, use only few for initialization
            inds_init = randperm(I);
            inds_init = inds_init(1:min(I,mopt.K*200));
            observations1 = observations(inds_init);

            
            % start keeping track of the execution time for this run
            mopt.start_time = tic;

            % initialize the parameters
            h3m = initialize_h3m_c(observations1,mopt);
            npar = 2*N;
            for j = 1 : K
                for i = 1 : I 
                    h3m.hmm{j}.qcov{i} = eye(npar); %[initial] proposal covariaance
                end
            end
            

            % if there are too many observations, use only some
            
            
            if mopt.max_obs < I
                observation_use = observations(1:mopt.max_obs);
                h3m_new = em_h3m_c_step(h3m,observation_use,mopt);
                h3m_new.init_h3m = h3m;
                h3m_new.time_elapsed_one = toc(mopt.start_time);
            else
                h3m_new = em_h3m_c_step(h3m,observations,mopt);
                h3m_new.init_h3m = h3m;
                h3m_new.time_elapsed_one = toc(mopt.start_time);
            end

            
            fprintf('Trial %d\t - loglikelihood: %d\n',t,h3m_new.LogL)

            if t == 1
                LL_best = h3m_new.LogL;
                h3m_out = h3m_new;
                t_best = t;
            elseif h3m_new.LogL > LL_best
                LL_best = h3m_new.LogL;
                h3m_out = h3m_new;
                t_best = t;
            end


        end
        
        if (re_do <= 11) &&((LL_best == -inf) || (isnan(LL_best)))
            fprintf('Need to do again... the LL was NaN ...\n')
            re_do = re_do + 1 ;
        end

    % Yuki: terminated the loop if re_do > 11
        if re_do > 11
            disp('nonononono')
            break
        end
        
    end
    
    MaxIter_time = h3m.MaxIter_time;
    
    fprintf('\nBest run is %d\n\n',t_best)
    
end

