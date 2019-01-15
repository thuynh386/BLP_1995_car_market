% This file estimates the demand using Nested Logit model
% There are 4 specifications: 
% OLS with/without IVs with/without brand fixed-effects
% There are also joint weak IV test and over-identification test.

global X Y IV1 IV2 IV3 Bdum 
Bdum1 = Bdum(:,[1:11 13:17]);

%% Nested logit estimation
% We have 2 endogenous variables and 2 sets of instruments.
% What if we just combine them all into 1 matrix???

Xn = [X car.lns];
Wn = horzcat(Xn(:,1:(end-2)),IV1,IV2,IV3);

Xd = horzcat(Bdum1,Xn);
Wd = horzcat(Xd(:,1:end-2),IV1,IV2,IV3);

%% (1) OLS without brand fixed-effects
[b1,b1_int] = regress(Y,Xn);
b1_SE = (b1_int(:,2) - b1) / norminv(0.975);
clear b1_int

%% (2) OLS brand fixed-effects
[b2,b2_int] = regress(Y,Xd);
b2 = b2(size(Bdum1,2)+1:end,:);
b2_int = b2_int(size(Bdum1,2)+1:end,:);
b2_SE = (b2_int(:,2) - b2) / norminv(0.975);
clear b2_int


%% (3) IV without brand fixed-effects
[b3,b3_SE] = tsls(Y,Xn,Wn);


%% (4) IV with brand fixed-effects
% Check again how to run regression with 2 endogenous variables
% especially when for each, there are multiple instruments
[b4,b4_SE] = tsls(Y,Xd,Wd);
b4 = b4(size(Bdum1,2)+1:end,:);
b4_SE = b4_SE(size(Bdum1,2)+1:end,:);

% csvwrite('nested.csv',[b1 b2 b3 b4 b1_SE b2_SE b3_SE b4_SE])

%% Instrument Tests
% Test for within-nest shares
T = Xn(:,end); % within-nest shares
W = X(:,1:end-2); % exogenous variables

Z = IV3(:,[12 13]); % horzcat(IV3,IV2,IV1); % Try with different IV sets
[beta se stats] = ivreg(Y,T,Z,W);

% Test for price
T = Xn(:,end-1); % price
W = X(:,1:end-2); % exogenous variables

Z = IV3(:,[12 13]); % horzcat(IV3,IV2,IV1); % Try with different IV sets
[beta se stats] = ivreg(Y,T,Z,W);

