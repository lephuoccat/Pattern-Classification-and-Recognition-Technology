%% Problem 2
close all 
clc, clear
% Import data from dataSet1.csv file
filename = "dataSet4.csv";
data = csvread(filename);                       % read data file
label = data(:,1);
X = data(:,2:3);
data0 = data(label==0,2:3);
data1 = data(label==1,2:3);

M = 2;
L = length(data)/M;
datatest = X(1:L,:);
labeltest = label(1:L,:);
datatrain = X(L+1:end,:);
labeltrain = label(L+1:end,:);

%---------
% part (a)
% LDA Model
% [YY, WW, lambdaX] = LDA(X, label);
MdlLinear = fitcdiscr(datatrain,labeltrain,'DiscrimType','linear');
[decision,score1,cost] = predict(MdlLinear,datatest);
[Pf1,Pd1] = ROC_curve(labeltest,score1(:,2),"all");
figure(1), set(gcf, 'Position',  [400, 0, 600, 600])
hold on, plot(Pf1,Pd1,'LineWidth',2)
hold on, grid on
xlabel('Probability of False Alarm (P_F)','FontSize',14) 
ylabel('Probability of Detection (P_D)','FontSize',14)
title(['ROC Curve for Data Set 4'],'FontSize',14) 

%---------
% part (b)
glm = fitglm(X,label,'linear','Distribution','binomial');
% prediction
score2 = predict(glm,datatest);
[Pf2,Pd2] = ROC_curve(labeltest,score2,"all");
hold on, plot(Pf2,Pd2,'LineWidth',2)

%---------
% part (c)
datatrain0 = datatrain(labeltrain==0,1:2);
datatrain1 = datatrain(labeltrain==1,1:2);

m_class0 = mean(datatrain0)';
m_class1 = mean(datatrain1)';
cov_class0 = cov(datatrain0);
cov_class1 = cov(datatrain1);

datatest = datatest';
% The general Bayes model
g0 = zeros(1,length(datatest));
g1 = zeros(1,length(datatest));
score3 = zeros(1,length(datatest));
C0 = inv(cov_class0);
C1 = inv(cov_class1);
iteration = length(datatest);
for i = 1:iteration
    g0(i) = datatest(:,i)'*(-0.5.*C0)*datatest(:,i) + (C0*m_class0)' * datatest(:,i) + ...
        (-0.5.*m_class0'*C0*m_class0 - log(det(cov_class0)) + log(0.5));
    g1(i) = datatest(:,i)'*(-0.5.*C1)*datatest(:,i) + (C1*m_class1)' * datatest(:,i) + ...
        (-0.5.*m_class1'*C1*m_class1 - log(det(cov_class1)) + log(0.5));
    score3(i) = (g1(i) - g0(i));
end

[Pf3,Pd3] = ROC_curve(labeltest,score3',"all");
hold on, plot(Pf3,Pd3,'LineWidth',2)
legend('LDA','Logistic Regression','Bayes');








% -------------------
% ROC Curve Function
% -------------------
function [X,Y] = ROC_curve(label_trains,statistics,threshold)
N = length(statistics);                         % length of the dataset
label_train_1 = sum(label_trains);              % number of H1 decision
label_train_0 = N - label_train_1;              % number of H0 decision
lambda = sort(statistics);                      % sort the decision statistics

% Choose the threshold criteria
if threshold == "all"                           % all decision staticstic as threshold
%     fprintf("Case (1): all\t\t");
    T = [-Inf; lambda; Inf];
    
elseif threshold == "linear"                    % linear sample threshold
    fprintf("Case (2): linear\t");
    min = lambda(1);
    max = lambda(N);
    T = [-Inf, linspace(min,max,99), Inf];
 
    
elseif threshold == "sample"                    % every nth sample threshold
    fprintf("Case (3): sample\t");
    if N >= 99
        n = floor(N/99);
        T = [-Inf; lambda(1:n:99*n,1); Inf];
    else
        n = 1;
        T = [-Inf; lambda(1:n:N,1); Inf];
    end
    
else
    lambda_h0 = zeros(label_train_0,1);         % decision statistics with label_train H0   
    index = 1;
    for i = 1:1:N
        if label_trains(i) == 0
            lambda_h0(index) = statistics(i);
            index = index + 1;
        end
    end
    
    if threshold == "H0"                        % every H0 decision statistics as threshold
        fprintf("Case (4): H0\t\t");
        T = [-Inf; sort(lambda_h0); Inf];
        
    elseif threshold == "interval"              % linearly sample P_f at interval 0.01
        fprintf("Case (5): interval\t");
        h0_sorted = sort(lambda_h0);
        if label_train_0 >= 100
            n = floor(label_train_0*0.01);
        else
            n = 1;
        end
        T = [h0_sorted(1:n:label_train_0); Inf];
    end
end
% fprintf("length of threshold: %d\n", length(T));

% Calculate prob of detection (Y) and
% prob of false alarm (X)
X = zeros(length(T),1);                         % x coor_traindinate or P_f
Y = zeros(length(T),1);                         % y coor_traindinate or P_d
for i = 1:1:length(T)
    count_d = 0;
    count_f = 0;
    for j = 1:1:N
        if statistics(j) >= T(i)                % if decision statistic > threshold
            if label_trains(j) == 1
                count_d = count_d + 1;          % add 1 to the H1 decision statistic
            else
                count_f = count_f + 1;          % add 1 to the H0 decision statistic
            end
        end
    end
    X(i) = count_f/label_train_0;               % Pf = (#H0 decision stats>threshold)/total H0 decision stats
    Y(i) = count_d/label_train_1;               % Pd = (#H1 decision stats>threshold)/total H1 decision stats
end    
end