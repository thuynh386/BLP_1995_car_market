% Main file for the analysis

%% Data import
data_import

%% Global variables / indices
global nprd varnames
nprd = 501;


%% Data clean and new variables creation
% Merge data and adjust for CPI index
USCPI.pindex = 100 ./ USCPI.cpi;
car = join(carpanel,USCPI(:,[1 3]));
car = join(car,gasprice);
car = join(car,UShouseholds);
car.dpm = car.gasprice .* car.pindex ./ car.mpg;
car.adp = car.p .* car.pindex / 1000; % adjusted price in thousands

% Dummy for number of doors
car.door3 = (car.dr==3);
car.door4 = (car.dr==4);
car.door5 = (car.dr==5);

% Market shares and outside option share
car.share_p = car.q ./ (car.nb_hh * 1000);

myear = [1977 1978 1979 1980 1981];
myear = repmat(myear,nprd,1);
outside = (repmat(car.year,1,5) == myear); % 501 x 5
outside_s = ones(5,1) - (outside' * car.share_p);
car.os = outside * outside_s;

clear myear outside_s outside carpanel gasprice USCPI UShouseholds


%% Matrices for regressions
global X Y C Bdum IV1 IV2 IV3
Y = log(car.share_p) - log(car.os);

% Vectors of car characteristics and adjusted price
C = car(:,{'dpm','door3','door4','door5','at','ps','air','drv','wt','hp2wt','hp','euro','japan','wb','size','adp'});
% Unit vector
U = array2table(ones([nprd,1]));

X = [U C];
X.Properties.VariableNames(1,1) = {'const'};
varnames = X.Properties.VariableNames';
X = table2array(X);


%% Generation of 'Brand' dummies
B = (unique(car.firmids))';
B = repmat(B,nprd,1);
B1 = repmat(car.firmids,1,size(B,2));
Bdum = (B == B1);
Bdum = 1 * Bdum; % To make it 'double' instead of 'logical'
Bdum = Bdum(:,2:end); % Reference: Firm with id '1'
clear B B1


%% Generation of IVs
niv = size(C,2) - 1;
ny = size(unique(car.year),1);
FID = repmat(unique(car.firmids)',nprd,ny);
nf = size(unique(car.firmids),1);
YID = [repmat(1977,nprd,nf) repmat(1978,nprd,nf) repmat(1979,nprd,nf) repmat(1980,nprd,nf) repmat(1981,nprd,nf)];
FID = (FID == repmat(car.firmids,1,(ny*nf))); % 501 x 95
YID = (YID == repmat(car.year,1,(ny*nf))); % 501 x 95


%% Characteristics of other products from the same firm as IV1
FYID = (FID .* YID);
ssf = table2array(C(:,1:niv))' * FYID;
ssf = ssf'; % 95 x 15

IV1 = zeros(nprd,niv); % 501 x 15

for i = 1:nprd
   index = find(FYID(i,:));
   IV1(i,:) = ssf(index,:);
end

IV1 = IV1 - table2array(C(:,1:niv));
clear nf ny ssf


%% Characteristics of products from competing firms as IV2
ssf = table2array(C(:,1:niv))' * ((1 - FID) .* YID);
ssf = ssf'; % 95 x 15

IV2 = zeros(nprd,niv); % 501 x 15

for i = 1:nprd
   index = find(FYID(i,:));
   IV2(i,:) = ssf(index,:);
end

IV2 = IV2 - table2array(C(:,1:niv)); % 501 x 15
clear FYID FID YID ssf i index


%% Calculate within-nest shares
% BS: Brand size; BY: Brand year; NS: Nest share; NQ: Nest quantity
BS = [repmat({'compact'},nprd,5) repmat({'midsize'},nprd,5) repmat({'large'},nprd,5)];
BS1 = repmat(car.cat,1,size(BS,2));
BS = (BS == BS1); % matrix 501 x 15

BY = repmat(unique(car.year)',nprd,3);
BY1 = repmat(car.year,1,size(BY,2));
BY = (BY == BY1); % matrix 501 x 15

% matrix 501x15 indexing for size category and year
NS = BS .* BY; 
% Vector of total quantity sold for each year * size category
NQ = (car.q)' * (BS .* BY); 

% Share within the nest, in percentage term
for i = 1:size(car.q,1)
    index = find(NS(i,:));
    car.ns(i) = car.q(i) / NQ(index);
end

car.lns = log(car.ns); % log of shares within nests
clear BS BS1 BY BY1 NS NQ


%% New instrumental variables for nested logit model (IV3)
% Sum of characteristics of products in the same nest in the same market
% minus own characteristics
niv = size(C,2) - 1; % number of instruments
ny = size(unique(car.year),1); % number of year
NID = repmat(unique(car.cat)',nprd,ny);
ng = size(unique(car.cat),1); % number of categories / nests
YID = [repmat(1977,nprd,ng) repmat(1978,nprd,ng) repmat(1979,nprd,ng) repmat(1980,nprd,ng) repmat(1981,nprd,ng)];
NID = (NID == repmat(car.cat,1,(ny*ng))); % 501 x 15
YID = (YID == repmat(car.year,1,(ny*ng))); % 501 x 15

NYID = (NID .* YID);
ssf = table2array(C(:,1:niv))' * NYID;
ssf = ssf'; % 15 x 15

IV3 = zeros(nprd,niv); % 501 x 15

for i = 1:nprd
   index = find(NYID(i,:));
   IV3(i,:) = ssf(index,:);
end

IV3 = IV3 - table2array(C(:,1:niv));
clear index ng ny ssf NID YID NYID niv i


%% Logit model
logit_model;


%% Nested logit model
nested_logit;


%% Random-coefficients model
random_coeff;
