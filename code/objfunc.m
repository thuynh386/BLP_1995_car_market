function f = objfunc(sigma)

global X1 X2 Z_ran mu taste s_actual nprd ns

% The mu component, for each product and each consumer
mu = (X2 * taste') * sigma; % matrix of 501 x 100
% mu = mu .* repmat(sigma,size(mu,1),size(mu,2));

% Initial guess for delta
delta = ones(nprd,1);
diffe = ones(nprd,1);

% iter = 0; 
while norm(diffe) > 10^(-9)
    mv = repmat(delta,1,ns) + mu;
    mv = exp(mv);
    sum_mv = sum(mv,1);
    sum_mv = repmat(sum_mv,nprd,1);
    % For each consumer and each product, the probability of buying
    prob = mv ./ (1 + sum_mv); % see the slides
    predicted_share = sum(prob,2); % matrix 501 x 1
    diffe = s_actual - predicted_share;
    delta = delta + log(s_actual) - log(predicted_share);
    % iter = iter + 1;
end

proj_ins = Z_ran * inv([Z_ran' * Z_ran]) * Z_ran';

if max(isnan(delta)) == 1
	f = 20e+10;
else
% 2SLS to estimate alpha, beta from the obtained delta
% Residuals are contained in 'resid'
    theta1 = inv(X1' * proj_ins * X1) * X1' * proj_ins * delta;
    resid = delta - X1 * theta1;
	f = resid' * proj_ins * resid;
    save resid
% GMM objective function
end 