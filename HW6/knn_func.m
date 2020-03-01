function [predicted_labels,nn_index,score] = knn_func(k,data,labels,sigma,t_data)

if nargin < 4
    error('Too few input arguments.')
end

if size(data,2)~=size(t_data,2)
    error('data should have the same dimensionality');
end

if mod(k,2)==0
    error('to reduce the chance of ties, please choose odd k');
end

%initialization
predicted_labels=zeros(size(t_data,1),1);
score=zeros(size(t_data,1),1);
ed=zeros(size(t_data,1),size(data,1));      %ed: (MxN) euclidean distances 
ind=zeros(size(t_data,1),size(data,1));     %corresponding indices (MxN)
k_nn=zeros(size(t_data,1),k);               %k-nearest neighbors for testing sample (Mxk)

%calc euclidean distances between each testing data point and the training
%data samples
for test_point=1:size(t_data,1)
    for train_point=1:size(data,1)
        S = [sigma(train_point,1:2);sigma(train_point,3:4)];
        %calc and store sorted euclidean distances with corresponding indices
        ed(test_point,train_point)=(t_data(test_point,:)-data(train_point,:))*S*...
            (t_data(test_point,:)-data(train_point,:))';
    end
    [ed(test_point,:),ind(test_point,:)]=sort(ed(test_point,:));
end

%find the nearest k for each data point of the testing data
k_nn=ind(:,1:k);
nn_index=k_nn(:,1);
%get the majority vote 
for i=1:size(k_nn,1)
    options=unique(labels(k_nn(i,:)'));
    max_count=0;
    max_label=0;
    for j=1:length(options)
        L=length(find(labels(k_nn(i,:)')==options(j)));
        if L>max_count
            max_label=options(j);
            max_count=L;
        end
    end
    predicted_labels(i)=max_label;
    score(i) = sum(labels(k_nn(i,:)'))/k;
end

