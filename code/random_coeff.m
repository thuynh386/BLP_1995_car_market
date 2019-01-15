%
% Random-coefficients logit model for car size

%% Matrix of products and markets. 192 products and 5 markets (years)
% prd_names = unique(car2.name);
% mkt_names = [1977 1978 1979 1980 1981];
% 
% % Matrix of shares for each product in each market
% prd_mkt = zeros(size(prd_names,1),size(mkt_names,2));
% for i = 1:size(prd_mkt,1)
%    for j = 1:size(prd_mkt,2)
%        prd_mkt(i,j) = transpose(car2.share_p) * (car2.name == prd_names(i,1) & car2.year == mkt_names(1,j));
%    end
% end

%% Random-coefficients model estimation
% What is the instrumental variable for 'size', the continuous variable

global X X1 X2 IV1 IV2 Z_ran ns s_actual taste
X1 = X(:,1:(end-2));
X2 = X(:,(end-1));
Z_ran = horzcat(X1,IV1,IV2);
ns = 100; % number of simulated individuals
s_actual = car.share_p;


%% Estimation steps:
% 'theta': alpha, beta
% 'theta2': sigma_eta
% Step 0: Draw 'eta' from N(0,1) then multiply by 'sigma_eta'
% Step 1: Given 'sigma_eta' and 'delta', compute predicted market shares as a
% function (loop 1)
% Step 2: Given 'sigma_eta', solve for 'delta' (inside loop 1)
%%%%%% Step 2b: Calculate the cost unobservables %%%%%%
% Step 3: Given 'theta', compute the error term, interact it with the
% instruments and the compute the value of objective function (loop 2)
% Step 4: Search for 'theta' minimizing the objective function (loop 2)

% Step 0: The random taste coefficients for size
rng(123456);
taste = normrnd(0,1,ns,1);
% taste = repmat(taste,1,5);

%Z_ran = horzcat(X1,IV1);
%Z_ran = horzcat(X1,IV2);


%% 1st method: Iterate until the convergence of sigma 
% Initial guess for variance of taste for size
options = optimoptions(@fminunc,'Algorithm','quasi-newton');

sigma_0 = 0;
sigma = 1.0; % Initial guess
iter = 0;
while norm(sigma - sigma_0) > 10^(-6)
    sigma_0 = (sigma_0 + sigma) / 2;
    [sigma,fval,exitflag,output] = fminunc(@objfunc,sigma_0,options);
    iter = iter + 1;
end


% %% 2nd method: compute the objective function value for a range of sigma
% theta2 = [0.1:0.1:10];
% for i=1:size(theta2,2)
%     sigma = theta2(i);
%     gmm(i) = objfunc(sigma);
% end
% plot(theta2,gmm);
% [m,i] = min(gmm);
