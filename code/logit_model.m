% This file estimates the demand using Logit model
% There are 4 specifications: 
% OLS with/without IVs with/without brand fixed-effects

global X Y IV1 IV2 Bdum 

W = horzcat(X(:,1:end-1),IV1,IV2);
Xd = horzcat(Bdum,X);
Wd = horzcat(Xd(:,1:end-1),IV1,IV2);

%% (1) OLS without IVs, without brand fixed-effects
[b1,b1_int] = regress(Y,X);
b1_SE = (b1_int(:,2) - b1) / norminv(0.975);
clear b1_int

%% (2) OLS without IVs, with brand fixed-effects
[b2,b2_int] = regress(Y,Xd);
b2 = b2(size(Bdum,2)+1:end,:);
b2_int = b2_int(size(Bdum,2)+1:end,:);
b2_SE = (b2_int(:,2) - b2) / norminv(0.975);
clear b2_int


%% (3) OLS with IVs, without brand fixed-effects
[b3,b3_SE] = tsls(Y,X,W);

%% Check correlation between corresponding IVs
for i=1:size(IV1,2)
    iv_cor(i) = corr(IV1(:,i),IV2(:,i));
end

% High correlation for: 'door4','wt','hp2wt','hp','wb','size'
IV11 = IV1(:,[3 9:11 14:15]);
W1 = horzcat(X(:,1:end-1),IV11,IV2);
[b31,b31_SE]= tsls(Y,X,W1);


%% (4) OLS with IVs, with brand fixed-effects
[b4,b4_SE] = tsls(Y,Xd,Wd);
b4 = b4(size(Bdum,2)+1:end,:);
b4_SE = b4_SE(size(Bdum,2)+1:end,:);


%% Correction for brand dummies
% Xd2 is to correct for rank defficiency of X3, removing firm 12 and 18
Bdum1 = Bdum(:,[1:11 13:17]);
Xd2 = horzcat(Bdum1,X);
Wd2 = horzcat(Xd2(:,1:end-1),IV1,IV2);

[b22,b22_int] = regress(Y,Xd2);
b22 = b22(size(Bdum1,2)+1:end,:);
b22_int = b22_int(size(Bdum1,2)+1:end,:);
b22_SE = (b22_int(:,2) - b22) / norminv(0.975);
clear b22_int


%% Test with different IV sets
Xd1 = horzcat(Bdum1,X);
Wd1 = horzcat(Xd1(:,1:end-1),IV1);
Wd2 = horzcat(Xd1(:,1:end-1),IV2);
Wd3 = horzcat(Xd1(:,1:end-1),IV1,IV2);

% Using only IV1
[b41,b41_SE] = tsls(Y,Xd1,Wd3);
b41 = b41(size(Bdum1,2)+1:end,:);
b41_SE = b41_SE(size(Bdum1,2)+1:end,:);
% Using only IV2
[b42,b42_SE] = tsls(Y,Xd1,Wd1);
b42 = b42(size(Bdum1,2)+1:end,:);
b42_SE = b42_SE(size(Bdum1,2)+1:end,:);
% Using both IV1 and IV2
[b43,b43_SE] = tsls(Y,Xd1,Wd2);
b43 = b43(size(Bdum1,2)+1:end,:);
b43_SE = b43_SE(size(Bdum1,2)+1:end,:);

clear W Xd Xd2 Wd2 Xd1 Wd1 Wd2 Wd3

% csvwrite('logit2.csv',[b1 b2 b3 b41 b42 b43]);
% csvwrite('logit2SE.csv',[b1_SE b2_SE b3_SE b41_SE b42_SE b43_SE]);


%% Instrument testing
% Using 'ivreg' routine

T = X(:,end); % endogenous variable
Z = horzcat(IV1,IV2); % instruments
W = X(:,1:end-1); % exogenous variables

[beta se stats] = ivreg(Y,T,IV1,W);
[beta se stats] = ivreg(Y,T,IV2,W);
[beta se stats] = ivreg(Y,T,Z,W);

% Experiment with some new sets of IVs
Z = IV1(:,[1 5:11 14:15]); % no door dummies, no origin dummies
[beta se stats] = ivreg(Y,T,Z,W);
Z = IV1(:,[1 8 9 11 15]); % only dpm, drv, wt, hp, size
[beta se stats] = ivreg(Y,T,Z,W);

Z = IV1(:,[12 13]); % only origin dummies: euro, japan
[beta se stats] = ivreg(Y,T,Z,W);

Z = IV2(:,[12 13]); % only origin dummies: euro, japan
[beta se stats] = ivreg(Y,T,Z,W);

%% Estimation with IV: euro, japan

W = horzcat(X(:,1:end-1),IV1(:,[12 13]));
Xd = horzcat(Bdum1,X);
Wd = horzcat(Xd(:,1:end-1),IV1(:,[12 13]));

[bori1,bori1_SE] = tsls(Y,X,W);
[bori12,bori12_SE] = tsls(Y,Xd,Wd);
bori12 = bori12(size(Bdum1,2)+1:end,:);
bori12_SE = bori12_SE(size(Bdum1,2)+1:end,:);
clear W Xd Wd
% [bori1 bori12]

W = horzcat(X(:,1:end-1),IV2(:,[12 13]));
Xd = horzcat(Bdum1,X);
Wd = horzcat(Xd(:,1:end-1),IV2(:,[12 13]));

[bori2,bori2_SE] = tsls(Y,X,W);
[bori22,bori22_SE] = tsls(Y,Xd,Wd);
bori22 = bori22(size(Bdum1,2)+1:end,:);
bori22_SE = bori22_SE(size(Bdum1,2)+1:end,:);
clear W Xd Wd
% [bori2 bori22]