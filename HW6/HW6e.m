%% Problem 7
close all 
clc, clear
% Import data from dataSetRingsWithSpeckle.csv file
filename = "dataSetRingsWithSpeckle.csv";
data = csvread(filename);                       % read data file
labels = data(:,1);
X = data(:,2:3);
data0 = data(labels==0,2:3);
data1 = data(labels==1,2:3);

% KNN Model
k = 7;
KNN = fitcknn(X,labels,'NumNeighbors',k,'Distance','euclidean');

% Create a grid of points spanning the entire space
x1 = linspace(-15,15, 251);
x2 = linspace(-15,15, 251);
[xTest1, xTest2] = meshgrid(x1,x2);
xTest = [xTest1(:) xTest2(:)];
% prediction
[decision,score,cost] = predict(KNN,xTest);
dsTest = reshape(score(:,2), length(x2), length(x1));
boundary1 = reshape(decision, length(x2), length(x1));

% Plot the decision statistic surface
figure(), set(gcf, 'Position',  [350, 0, 600, 600])
hold on, imagesc(x1([1 end]), x2([1 end]), dsTest);
colorbar
    
% plot data
hold on, plot(data0(:,1),data0(:,2),'ko','LineWidth',2);
hold on, plot(data1(:,1),data1(:,2),'r*','LineWidth',2);
hold on, contour(x1, x2, boundary1,'k-','LineWidth',2);
xlabel('Feature 1','FontSize',14) 
ylabel('Feature 2','FontSize',14)
title('Data Set Rings With Speckle','FontSize',14)
legend('Class 0','Class 1','KNN k=7')
axis([-15 15 -15 15]);

%%
% DANN
N = 100;
I = eye(2,2);
epsilon = 1;

sigma = zeros(length(data),4);
for i = 1:length(data)
    x0 = X(i,:);
    L2 = zeros(length(data),1);
    for j = 1: length(data)
        L2(j) = norm(X(j,:) - x0);
    end
    L2_data = [L2,data];
    [~,idx] = sort(L2_data(:,1));
    sorted_L2 = L2_data(idx,:);
    
    X_NN = sorted_L2(1:N,3:4);
    label = sorted_L2(1:N,2);
    MdlLinear = fitcdiscr(X_NN,label,'DiscrimType','linear');
    W = MdlLinear.Sigma;
    B = MdlLinear.BetweenSigma;
    S = W^(-0.5)*( W^(-0.5)*B*W^(-0.5) + epsilon.*I )*W^(-0.5);
    sigma(i,:) = [S(1,:),S(2,:)];
end
% DANN model
[decision,nn,score] = knn_func(k,X,labels,sigma,xTest);

% Plot
dsTest = reshape(score, length(x2), length(x1));
boundary2 = reshape(decision, length(x2), length(x1));
figure()
set(gcf, 'Position',  [350, 0, 800, 600])
hold on, imagesc(x1([1 end]), x2([1 end]), dsTest);
colorbar

% plot data
hold on, plot(data0(:,1),data0(:,2),'ko','LineWidth',2);
hold on, plot(data1(:,1),data1(:,2),'r*','LineWidth',2);
hold on, contour(x1, x2, boundary2,'k-','LineWidth',2);
xlabel('Feature 1','FontSize',14) 
ylabel('Feature 2','FontSize',14)
title('Data Set Rings With Speckle','FontSize',14)
legend('Class 0','Class 1','DANN k=7 N=100')
axis([-15 15 -15 15]);

%%
% plot data
figure()
set(gcf, 'Position',  [350, 0, 800, 600])
hold on, plot(data0(:,1),data0(:,2),'ko','LineWidth',2);
hold on, plot(data1(:,1),data1(:,2),'r*','LineWidth',2);
hold on, contour(x1, x2, boundary1,'c-','LineWidth',2);
hold on, contour(x1, x2, boundary2,'b-','LineWidth',2);
xlabel('Feature 1','FontSize',14) 
ylabel('Feature 2','FontSize',14)
title('Data Set Rings With Speckle','FontSize',14)
legend('Class 0','Class 1','KNN k=7','DANN k=7 N=100')
axis([-15 15 -15 15]);
