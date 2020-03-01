%% Problem 3
close all 
clc, clear
% Import data from dataSet2.csv file
filename = "dataSet2.csv";
data = csvread(filename);                       % read data file
label = data(:,1);
X = data(:,2:3);
data0 = data(label==0,2:3);
data1 = data(label==1,2:3);

%---------
% part (a)
% LDA Model
% [YY, WW, lambdaX] = LDA(X, label);
MdlLinear = fitcdiscr(X,label,'DiscrimType','linear');

% Testing data decision statistics
x1 = linspace(-15, 20, 500);
x2 = linspace(-15, 15, 500);
[xTest1, xTest2] = meshgrid(x1,x2);
xTest = [xTest1(:) xTest2(:)];

% prediction
[decision,score,cost] = predict(MdlLinear,xTest);
dsTest1 = reshape(score(:,2), length(x2), length(x1));
figure()
set(gcf, 'Position',  [350, 0, 800, 600])
hold on, imagesc(x1([1 end]), x2([1 end]), dsTest1);
colorbar

% plot data
hold on, plot(data0(:,1),data0(:,2),'ko','LineWidth',2);
hold on, plot(data1(:,1),data1(:,2),'r*','LineWidth',2);
xlabel('Feature 1','FontSize',14) 
ylabel('Feature 2','FontSize',14)
title('Data Set 2','FontSize',14)
legend('Class 0','Class 1')
axis([-15 20 -15 15]);

% decision boundary
K = MdlLinear.Coeffs(1,2).Const;  
L = MdlLinear.Coeffs(1,2).Linear;
f1 = @(x1,x2) K + L(1)*x1 + L(2)*x2;
hold on
h1 = fimplicit(f1,[-15 20 -15 15]);
h1.Color = 'c';
h1.LineWidth = 3;
h1.DisplayName = 'LDA decision boundary';

%---------
% part (b)
% Y_label = categorical(label);
% [B,dev,stats] = mnrfit(X,Y_label);
glm = fitglm(X,label,'linear','Distribution','binomial');

% prediction
ypred = predict(glm,xTest);
dsTest2 = reshape(ypred, length(x2), length(x1));
figure()
set(gcf, 'Position',  [350, 0, 800, 600])
hold on, imagesc(x1([1 end]), x2([1 end]), dsTest2);
colorbar

% plot data
hold on, plot(data0(:,1),data0(:,2),'ko','LineWidth',2);
hold on, plot(data1(:,1),data1(:,2),'r*','LineWidth',2);
xlabel('Feature 1','FontSize',14) 
ylabel('Feature 2','FontSize',14)
title('Data Set 2','FontSize',14)
axis([-15 20 -15 15]);

% decision boundary
f2 = (dsTest2>=0.5);
hold on, contour(x1, x2, f2,'c','LineWidth',2);
legend('Class 0','Class 1','Logistic Discriminant decision boundary')


%---------
% part (c)
m_class0 = mean(data0)';
m_class1 = mean(data1)';
cov_class0 = cov(data0);
cov_class1 = cov(data1);
xTest = xTest';

% The general Bayes model
g0 = zeros(1,length(xTest));
g1 = zeros(1,length(xTest));
score = zeros(1,length(xTest));
C0 = inv(cov_class0);
C1 = inv(cov_class1);
iteration = length(xTest);
for i = 1:iteration
    g0(i) = xTest(:,i)'*(-0.5.*C0)*xTest(:,i) + (C0*m_class0)' * xTest(:,i) + ...
        (-0.5.*m_class0'*C0*m_class0 - log(det(cov_class0)) + log(0.5));
    g1(i) = xTest(:,i)'*(-0.5.*C1)*xTest(:,i) + (C1*m_class1)' * xTest(:,i) + ...
        (-0.5.*m_class1'*C1*m_class1 - log(det(cov_class1)) + log(0.5));
    score(i) = (g1(i) >= g0(i));
end
dsTest3 = reshape(score, length(x2), length(x1));

% Plot the decision statistic surface
figure()
set(gcf, 'Position',  [350, 0, 800, 600])
hold on, imagesc(x1([1 end]), x2([1 end]), dsTest3);
colorbar
title('General Bayes Classifier')
axis([-15 20 -15 15]);

% plot data
hold on, plot(data0(:,1),data0(:,2),'ko','LineWidth',2);
hold on, plot(data1(:,1),data1(:,2),'r*','LineWidth',2);
xlabel('Feature 1','FontSize',14) 
ylabel('Feature 2','FontSize',14)
title('Data Set 2','FontSize',14)
axis([-15 20 -15 15]);

% decision boundary
f3 = (dsTest3>=0.5);
hold on, contour(x1, x2, f3,'c','LineWidth',2);
legend('Class 0','Class 1','Bayes Classifier decision boundary')

% Plot the boundaries
% plot data
figure()
set(gcf, 'Position',  [350, 0, 800, 600])
hold on, plot(data0(:,1),data0(:,2),'ko','LineWidth',2);
hold on, plot(data1(:,1),data1(:,2),'r*','LineWidth',2);
xlabel('Feature 1','FontSize',14) 
ylabel('Feature 2','FontSize',14)
title('Data Set 2','FontSize',14)
axis([-15 20 -15 15]);

hold on
h1 = fimplicit(f1,[-15 20 -15 15]);
h1.Color = 'r';
h1.LineWidth = 3;
hold on, contour(x1, x2, f2,'b','LineWidth',2);
hold on, contour(x1, x2, f3,'c','LineWidth',2);
legend('Class 0','Class 1','LDA decision boundary','Logistic Discriminant decision boundary','Bayes Classifier decision boundary')
