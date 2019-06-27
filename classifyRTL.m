close all
clear all
clc


%% Load Data

load mnist_basic.mat;

% X_train= (Train.X)';
% label_train= Train.y;
% X_test= (Test.X)';
% label_test= Test.y;


% load usps.mat

% load banglatraindata.mat;
% load bangtestdata.mat;

% load Devnagritraindata.mat;
% load Devnagritestdata.mat;

% X_train= trn.X;
% label_train= trn.y;
% X_test= tst.X;
% label_test= tst.y;

% X_train = loadMNISTImages('train-images.idx3-ubyte'); % train_image_matrix
% label_train = loadMNISTLabels('train-labels.idx1-ubyte');    % train_labels_matrix
% X_test = loadMNISTImages('t10k-images.idx3-ubyte'); % test_image_matrix
% label_test = loadMNISTLabels('t10k-labels.idx1-ubyte'); %test_image_labels

%%
% adding noise
% X_train= imnoise(X_train,'gaussian');
%%
% Parameters

tau = 0.1;
mu = 1;
numOfAtoms= 300;

[T,Z_train] =  RobustTransformLearning (X_train, numOfAtoms, mu, tau);  


%% Testing

tic

Z_test = sign(T*X_test).*max(0,abs(T*X_test)-tau);   

toc
time= toc-tic;

%% Classification (KNN)

label = label_train;
testlabels = label_test;

if min(label_train) == 0
    label = label + 1;
    testlabels = testlabels + 1;
end

knnmodel = fitcknn (Z_train', label);
PredLabels = predict (knnmodel, Z_test');
 

disp('Out of Classifier!');
correct = length(find(PredLabels==testlabels))
percent =correct/length(testlabels) * 100

%%

% %construct a binary SVM classifier
opts = optimset('MaxIter',2000,'TolX',5e-4,'TolFun',5e-4);
label = label_train;
testlabels = label_test;

if min(label_train) == 0
    label = label + 1;
    testlabels = testlabels + 1;
end

svmModel = cell(1,max(label));

for ii = min(label):max(label)
    fprintf('Learning SVM classifier for Class %d ... ',ii)
    currentClass = Z_train(:,label==ii);
    negClass = Z_train(:,label~=ii);
    posLabel = 1*ones(1,size(currentClass,2));
    negLabel = -1*ones(1,size(negClass,2));
    Xtr = [currentClass,negClass];
    Ltr = [posLabel,negLabel];
    t = randperm(size(Xtr,2));
    Xtr = Xtr(:,t);
    Ltr = Ltr(1,t);
    svmModel{1,ii} = fitcsvm(Xtr',Ltr,'Standardize',true,'KernelFunction','rbf','KernelScale','auto');
    fprintf('SVM Learnt \n');
end

Scores = zeros(size(Z_test,2),numel(unique(label_test)));

for class = min(testlabels):max(testlabels)
    [~,score] = predict(svmModel{1,class},Z_test');
    Scores(:,class) = score(:,2);
end

[~,maxScore] = max(Scores,[],2); %maxScore is px1

if size(label_test,2) >1
    testlabels = testlabels';
end

accSVM = length(find((maxScore) == testlabels))/length(label_test);
fprintf('Accuracy = %d',accSVM*100)





