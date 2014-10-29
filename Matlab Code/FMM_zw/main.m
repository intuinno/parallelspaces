clear all
clc
%% main call of training and testing algorithm
CVNeed = 0;
IfDoTest = 1;

%% if cross validation is needed
if CVNeed == 1
    
nCV = 5;
cv_err = zeros(nCV,1);
for cv = 1:nCV
    load(['data/train_cv' num2str(cv) '.mat']);
    load(['data/test_cv' num2str(cv) '_train.mat']);
    load(['data/test_cv' num2str(cv) '_test.mat']);
    
    % pearson similarity
%     [model,cv_err(cv)] = Pearson(TrainSet,TestTrainSet,TestTestSet);
    
    % vector space similarity
%     cv_err(cv) = VecSim(TrainSet,TestTrainSet,TestTestSet);
    
    % FMM
%     cv_err(cv) = FMM(TrainSet,TestTrainSet,TestTestSet);
    
    % MF
%     cv_err(cv) = MF(TrainSet,TestTrainSet,TestTestSet);
end

end

%% use all data to train model
if IfDoTest == 1

WholeTrainData = load('data/train.txt');
% model = FMM_inference4(WholeTrainData);

ModelPreFix = 'data/FMM_Anneal_Retry/';
TestCase = [5,10,20];
UIDOffset = [200,300,400];
for tc = 1:length(TestCase)
    % test
    load(['data/test' num2str(TestCase(tc)) '_train']);
    load(['data/test' num2str(TestCase(tc)) '_test']);
    % pearson method
%     [model,PredVal,err] = Pearson(WholeTrainData,RealTestTrainData,RealTestTestData);
    % vector space model
    [model,PredVal,err] = VectorSpace(WholeTrainData,RealTestTrainData,RealTestTestData);
    % item based similarity method
%      [model,PredVal,err] = ItemSimilarityModel(WholeTrainData,RealTestTrainData,RealTestTestData);
    % Flexible Mixture Model
%     model = FMM_inference4(WholeTrainData);
%     PredVal = FMM_predict_final(model,WholeTrainData,RealTestTrainData,RealTestTestData);
    % MF 
%     model = pmf_train(WholeTrainData/5,50,1,WholeTrainData/5);
%     PredVal = pmf_test_final(RealTestTrainData/5,RealTestTestData/5,model);

    
    %
    fid = fopen([ModelPreFix 'result' num2str(TestCase(tc)) '.txt'],'wt');
    for uid = 1:size(RealTestTestData,1)
        for iid = 1:size(RealTestTestData,2)
            if RealTestTestData(uid,iid) ~= 0
                fprintf(fid,'%d %d %d\n',uid+UIDOffset(tc),iid,PredVal(uid,iid));
            end
        end
    end
    fclose(fid);
end


end





