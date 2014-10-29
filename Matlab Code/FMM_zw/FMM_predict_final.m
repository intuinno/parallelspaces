function PredVal = FMM_predict_final(model,TrainRateData,TestRateData,TestTestData)
%% speed up predict.m, in fold-in process, corresponding to FMM_inference2.m
prob_zx = model.prob_zx;    % prob of zx, nUserClass * 1
prob_zy = model.prob_zy;    % prob of zy, nItemClass * 1
prob_x_zx = model.prob_x_zx;    % prob of user x given zx, nUser * nUserClass
prob_y_zy = model.prob_y_zy;    % prob of item y give zy, nItem * nItemClass
prob_r_zxzy = model.prob_r_zxzy;    % prob of rate give zx and zy, nUserClass * nItemClass * nRate
nUserClass = length(prob_zx);
nItemClass = length(prob_zy);
nRate = size(prob_r_zxzy,3);
b = model.b;

[nTestUser,nItem]= size(TestRateData);
totalerr = zeros(nTestUser,1);
count = zeros(nTestUser,1);
for OUTERuid = 1:nTestUser
    if mod(OUTERuid,10) == 0
        fprintf('Testing the %d-th user\n',OUTERuid);
    end
    %% fold-in process by re-run EM algorithm 
    % pick some given items for testing users
    RatePos = find(TestRateData(OUTERuid,:)~=0);
%     nPickItem = fix(length(RatePos)/2);
%     TrainLen = RatePos(nPickItem);
%     TestLen = nItem - TrainLen;
%     CurUserTestTrainData = [TestRateData(OUTERuid,1:TrainLen) zeros(1,TestLen)];
    CurUserTestTrainData = TestRateData(OUTERuid,:);
    DefaultRate = mean(TestRateData(OUTERuid,RatePos));
%     CurUserTestTestData = [zeros(1,TrainLen) TestRateData(OUTERuid,(TrainLen+1):end)];
    CurUserTestTestData = TestTestData(OUTERuid,:);

    curRateData = [TrainRateData; CurUserTestTrainData];
    [nUser,~] = size(curRateData);
    
    % prob of user x given zx, by folding in new test user: (nUser+1) * nUserClass
    prob_test_x_zx = [prob_x_zx; rand(1,nUserClass)];
    prob_test_x_zx_sum = sum(prob_test_x_zx+1e-10,1);
    prob_test_x_zx = prob_test_x_zx ./ repmat(prob_test_x_zx_sum,nUser,1);
    
    %% re-run EM algorithm
    [userList,itemList] = find(curRateData~=0);
    rateList = curRateData(find(curRateData~=0));
    nInstance = length(userList);
    temp_prob_y_zy = zeros(1,nItemClass,nInstance);    temp_prob_y_zy(1,:,:) = prob_y_zy(itemList,:)';
    temp_cur_post_zx_zy = repmat(prob_zx,[1,nItemClass,nInstance]).*repmat(prob_zy',[nUserClass,1,nInstance])...
          .*repmat(temp_prob_y_zy,[nUserClass,1,1]).*prob_r_zxzy(:,:,rateList);
    userindex = cell(nUser,1);
    for r = 1:nUser
        userindex{r} = find(userList==r);
    end
      
    StopCriteria = 0.02;
    IterCount = 0;
    IterUB = 100;
    IterMeasure = inf;
    while IterMeasure>StopCriteria && IterCount<IterUB
        IterCount = IterCount + 1;
        %% E step: posterior probability    
        temp_prob_test_x_zx = zeros(nUserClass,1,nInstance);    temp_prob_test_x_zx(:,1,:) = prob_test_x_zx(userList,:)';
        cur_post_zx_zy = temp_cur_post_zx_zy.*repmat(temp_prob_test_x_zx,[1,nItemClass,1]);
        cur_post_zx_zy = cur_post_zx_zy.^b;
        cur_post_zx_zy_sum = sum(sum(cur_post_zx_zy+1e-10,1),2);
        cur_post_zx_zy = cur_post_zx_zy ./ repmat(cur_post_zx_zy_sum,[nUserClass,nItemClass,1]);
        
        %% M step: update
        % update prob_x_zx 
        cur_prob_test_x_zx = zeros(nUser,nUserClass);
        for r = 1:nUser
            index = userindex{r};
            temp_cur_prob_test_x_zx = sum(sum(cur_post_zx_zy(:,:,index),2),3);
            cur_prob_test_x_zx(r,:) = squeeze(temp_cur_prob_test_x_zx)';
        end

        %% normalize probability
        cur_prob_test_x_zx_norm = sum(cur_prob_test_x_zx+1e-10,1);
        cur_prob_test_x_zx = cur_prob_test_x_zx ./ repmat(cur_prob_test_x_zx_norm,nUser,1);


        %% measure change
        prob_test_x_zx_diff = cur_prob_test_x_zx - prob_test_x_zx;
        cg_prob_x_zx = norm(prob_test_x_zx_diff(:));
        IterMeasure = cg_prob_x_zx / norm(prob_test_x_zx(:));
        
        prob_test_x_zx = cur_prob_test_x_zx;
    end
    
    %% test the accuracy on remaining data
    TestTestRateList = find(CurUserTestTestData~=0);
    TTR_len = length(TestTestRateList);
    joint_prob_inst = zeros(nRate,TTR_len);
    for rid = 1:nRate
        prob_zxzy_inst = zeros(nUserClass,nItemClass,TTR_len);
        temp_prob_y_zy = zeros(1,nItemClass,TTR_len);   temp_prob_y_zy(1,:,:) = prob_y_zy(TestTestRateList,:)';
        temp_prob_r_zxzy = zeros(nUserClass,nItemClass,1);  temp_prob_r_zxzy(:,:,1) = prob_r_zxzy(:,:,rid);
        prob_zxzy_inst = repmat(prob_zx,[1,nItemClass,TTR_len]).*repmat(prob_zy',[nUserClass,1,TTR_len])...
            .*repmat(prob_test_x_zx(end,:)',[1,nItemClass,TTR_len]).*repmat(temp_prob_y_zy,[nUserClass,1,1])...
            .*repmat(temp_prob_r_zxzy,[1,1,TTR_len]);
        temp_joint_prob_inst = sum(sum(prob_zxzy_inst,1),2);
        joint_prob_inst(rid,:) = squeeze(temp_joint_prob_inst);
    end
    joint_prob_inst_sum = sum(joint_prob_inst,1);
    nonzero_index = (joint_prob_inst_sum~=0);
    temp_joint_prob_inst_sum = joint_prob_inst_sum.*nonzero_index + double(1-nonzero_index);
    
    expected_rate = ((1:nRate)*joint_prob_inst) ./ (temp_joint_prob_inst_sum+1e-10);
    expected_rate = expected_rate.*nonzero_index + DefaultRate*(1-nonzero_index);
    actual_rate = CurUserTestTestData(TestTestRateList);
    curerr = sum(abs(expected_rate-actual_rate));
    curcount = TTR_len;

    totalerr(OUTERuid) = curerr;
    count(OUTERuid) = curcount;

    fprintf('User: %d, local MAE is:%f\n',OUTERuid,curerr/curcount);
    
    PredVal(OUTERuid,TestTestRateList) = round(expected_rate);
    
end

% accuracy = sum(totalerr) / sum(count);
% fid = fopen('TestResult.txt','wt');
% fprintf(fid,'%f',accuracy);
% fclose(fid);


PredVal = (PredVal>5)*5 + (PredVal<1)*1 + (PredVal<=5).*(PredVal>=1).*PredVal;

