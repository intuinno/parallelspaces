function model = FMM_inference4(CpltTrainData)
%% speed up version of FMM_inference2.m (with annealed EM)
% split into training and validation set
[nTotaluser,~] = size(CpltTrainData);
TrainSize = max(fix(nTotaluser*0.9),nTotaluser-20);
TestSize = nTotaluser - TrainSize;
TrainData = CpltTrainData(1:TrainSize,:);
TestData = CpltTrainData((TrainSize+1):end,:);
[nUser,nItem] = size(TrainData);
fprintf('Train set size:%d, validation set size:%d\n',TrainSize,TestSize);

% latent class number
nUserClass = 20;
nItemClass = 30;
nRate = 5;

%% annealed EM algorithm
b = 1.2;% annealed EM parameter
valset_perf = 5;% performance on validation set
b_seq = [];
valset_perf_seq = [];
while b>0.9
    prob_zx = rand(nUserClass,1);   prob_zx = prob_zx / norm(prob_zx);
    prob_zy = rand(nItemClass,1);   prob_zy = prob_zy / norm(prob_zy);
    prob_x_zx = rand(nUser,nUserClass); prob_x_zx = prob_x_zx ./ repmat( sum(prob_x_zx,1), nUser,1  );
    prob_y_zy = rand(nItem,nItemClass); prob_y_zy = prob_y_zy ./ repmat( sum(prob_y_zy,1), nItem,1  );
    prob_r_zxzy = rand(nUserClass,nItemClass,nRate); prob_r_zxzy = prob_r_zxzy ./ repmat( sum(prob_r_zxzy,3) ,[1,1,nRate]);

    StopCriteria = 0.02;
    IterCount = 0;
    IterUB = 100;
    IterMeasure = inf;
    while IterMeasure>StopCriteria && IterCount<IterUB
        IterCount = IterCount + 1;

        %% E step: posterior probability
        [userList,itemList] = find(TrainData~=0);
        rateList = TrainData(find(TrainData~=0));
        nInstance = length(userList);

        temp_prob_x_zx = zeros(nUserClass,1,nInstance);    temp_prob_x_zx(:,1,:) = prob_x_zx(userList,:)';
        temp_prob_y_zy = zeros(1,nItemClass,nInstance);    temp_prob_y_zy(1,:,:) = prob_y_zy(itemList,:)';
        cur_post_zx_zy = repmat(prob_zx,[1,nItemClass,nInstance]).*repmat(prob_zy',[nUserClass,1,nInstance])...
            .*repmat(temp_prob_x_zx,[1,nItemClass,1]).*repmat(temp_prob_y_zy,[nUserClass,1,1])...
            .*prob_r_zxzy(:,:,rateList);
        cur_post_zx_zy = cur_post_zx_zy.^b;
        cur_post_zx_zy_sum = sum(sum(cur_post_zx_zy+1e-10,1),2);
        cur_post_zx_zy = cur_post_zx_zy ./ repmat(cur_post_zx_zy_sum,[nUserClass,nItemClass,1]);

        %% M step: update
        % update prob_zx
        cur_prob_zx = sum(sum(cur_post_zx_zy,2),3);

        % update prob_zy 
        cur_prob_zy = sum(sum(cur_post_zx_zy,1),3);
        cur_prob_zy = cur_prob_zy';

        % update prob_x_zx  
        cur_prob_x_zx = zeros(nUser,nUserClass);
        for r = 1:nUser
            index = find(userList==r);
            temp_cur_prob_x_zx = sum(sum(cur_post_zx_zy(:,:,index),2),3);
            cur_prob_x_zx(r,:) = squeeze(temp_cur_prob_x_zx)';
        end

        % update prob_y_zy
        cur_prob_y_zy = zeros(nItem,nItemClass);
        for r = 1:nItem
            index = find(itemList==r);
            temp_cur_prob_y_zy = sum(sum(cur_post_zx_zy(:,:,index),1),3);
            cur_prob_y_zy(r,:) = squeeze(temp_cur_prob_y_zy);
        end

        % update prob_r_zxzy
        cur_prob_r_zxzy = zeros(nUserClass,nItemClass,nRate);
        for r = 1:nRate
            index = find(rateList==r);
            temp_cur_prob_r_zxzy = sum(cur_post_zx_zy(:,:,index),3);
            cur_prob_r_zxzy(:,:,r) = temp_cur_prob_r_zxzy;
        end 


        %% normalize probability
        cur_prob_zx_norm = sum(cur_prob_zx);
        cur_prob_zx = cur_prob_zx / cur_prob_zx_norm;

        cur_prob_zy_norm = sum(cur_prob_zy);
        cur_prob_zy = cur_prob_zy / cur_prob_zy_norm;

        cur_prob_x_zx_norm = sum(cur_prob_x_zx+1e-10,1);
        cur_prob_x_zx = cur_prob_x_zx ./ repmat(cur_prob_x_zx_norm,nUser,1);

        cur_prob_y_zy_norm = sum(cur_prob_y_zy+1e-10,1);
        cur_prob_y_zy = cur_prob_y_zy ./ repmat(cur_prob_y_zy_norm,nItem,1);

        cur_prob_r_zxzy_norm = sum(cur_prob_r_zxzy+1e-10,3);
        cur_prob_r_zxzy = cur_prob_r_zxzy ./ repmat(cur_prob_r_zxzy_norm,[1,1,nRate]);


        %% measure change
        prob_zx_diff = cur_prob_zx - prob_zx;
        cg_prob_zx = norm(prob_zx_diff(:)) / norm(prob_zx(:));
        prob_zy_diff = cur_prob_zy - prob_zy;
        cg_prob_zy = norm(prob_zy_diff(:)) / norm(prob_zy(:));
        prob_x_zx_diff = cur_prob_x_zx - prob_x_zx;
        cg_prob_x_zx = norm(prob_x_zx_diff(:)) / norm(prob_x_zx(:));
        prob_y_zy_diff = cur_prob_y_zy - prob_y_zy;
        cg_prob_y_zy = norm(prob_y_zy_diff(:)) / norm(prob_y_zy(:));
        prob_r_zxzy_diff = cur_prob_r_zxzy - prob_r_zxzy;
        cg_prob_r_zxzy = norm(prob_r_zxzy_diff(:)) / norm(prob_r_zxzy(:));
        IterMeasure = max([cg_prob_zx cg_prob_zy cg_prob_x_zx cg_prob_y_zy cg_prob_r_zxzy]);

        fprintf('Training Iteration: %d, parameter change is: %f\n',IterCount,IterMeasure);

        prob_zx = cur_prob_zx;
        prob_zy = cur_prob_zy;
        prob_x_zx = cur_prob_x_zx;
        prob_y_zy = cur_prob_y_zy;
        prob_r_zxzy = cur_prob_r_zxzy;

        model.prob_zx = prob_zx;
        model.prob_zy = prob_zy;
        model.prob_x_zx = prob_x_zx;
        model.prob_y_zy = prob_y_zy;
        model.prob_r_zxzy = prob_r_zxzy;
        model.b = b;
    end
    %% measure performance on validation set
    cur_valset_perf = FMM_predict2(model,TrainData,TestData);
    fprintf('b:%f, MAE on val set:%f\n',b,cur_valset_perf);
    b_seq = [b_seq b];
    valset_perf_seq = [valset_perf_seq cur_valset_perf];
    b = b * 0.95;
end
fprintf('****************b val and valset perf*********************\n');
for x = 1:length(b_seq)
    fprintf('%f:%f\n',b_seq(x),valset_perf_seq(x));
end
[temp_Y,temp_I] = sort(valset_perf_seq,'ascend');
b = b_seq(temp_I(1));
fprintf('*********Select best b: %f**********\n',b);    

%% re-run on whole dataset with fixed b
fprintf('--------------------begin re-running em-------------------\n');
TrainData = CpltTrainData;
[nUser,nItem] = size(TrainData);
prob_zx = rand(nUserClass,1);   prob_zx = prob_zx / norm(prob_zx);
prob_zy = rand(nItemClass,1);   prob_zy = prob_zy / norm(prob_zy);
prob_x_zx = rand(nUser,nUserClass); prob_x_zx = prob_x_zx ./ repmat( sum(prob_x_zx,1), nUser,1  );
prob_y_zy = rand(nItem,nItemClass); prob_y_zy = prob_y_zy ./ repmat( sum(prob_y_zy,1), nItem,1  );
prob_r_zxzy = rand(nUserClass,nItemClass,nRate); prob_r_zxzy = prob_r_zxzy ./ repmat( sum(prob_r_zxzy,3) ,[1,1,nRate]);


StopCriteria = 0.02;
IterCount = 0;
IterUB = 100;
IterMeasure = inf;
while IterMeasure>StopCriteria && IterCount<IterUB
	IterCount = IterCount + 1;    
    %% E step: posterior probability
    [userList,itemList] = find(TrainData~=0);
    rateList = TrainData(find(TrainData~=0));
    nInstance = length(userList);

    temp_prob_x_zx = zeros(nUserClass,1,nInstance);    temp_prob_x_zx(:,1,:) = prob_x_zx(userList,:)';
    temp_prob_y_zy = zeros(1,nItemClass,nInstance);    temp_prob_y_zy(1,:,:) = prob_y_zy(itemList,:)';
    cur_post_zx_zy = repmat(prob_zx,[1,nItemClass,nInstance]).*repmat(prob_zy',[nUserClass,1,nInstance])...
        .*repmat(temp_prob_x_zx,[1,nItemClass,1]).*repmat(temp_prob_y_zy,[nUserClass,1,1])...
        .*prob_r_zxzy(:,:,rateList);
    cur_post_zx_zy = cur_post_zx_zy.^b;
    cur_post_zx_zy_sum = sum(sum(cur_post_zx_zy+1e-10,1),2);
    cur_post_zx_zy = cur_post_zx_zy ./ repmat(cur_post_zx_zy_sum,[nUserClass,nItemClass,1]);
    
    %% M step: update
    % update prob_zx
    cur_prob_zx = sum(sum(cur_post_zx_zy,2),3);
            
    % update prob_zy 
    cur_prob_zy = sum(sum(cur_post_zx_zy,1),3);
    cur_prob_zy = cur_prob_zy';
  
    % update prob_x_zx  
    cur_prob_x_zx = zeros(nUser,nUserClass);
    for r = 1:nUser
        index = find(userList==r);
        temp_cur_prob_x_zx = sum(sum(cur_post_zx_zy(:,:,index),2),3);
        cur_prob_x_zx(r,:) = squeeze(temp_cur_prob_x_zx)';
    end
            
    % update prob_y_zy
    cur_prob_y_zy = zeros(nItem,nItemClass);
    for r = 1:nItem
        index = find(itemList==r);
        temp_cur_prob_y_zy = sum(sum(cur_post_zx_zy(:,:,index),1),3);
        cur_prob_y_zy(r,:) = squeeze(temp_cur_prob_y_zy);
    end
            
    % update prob_r_zxzy
    cur_prob_r_zxzy = zeros(nUserClass,nItemClass,nRate);
    for r = 1:nRate
        index = find(rateList==r);
        temp_cur_prob_r_zxzy = sum(cur_post_zx_zy(:,:,index),3);
        cur_prob_r_zxzy(:,:,r) = temp_cur_prob_r_zxzy;
    end 
 
    
    %% normalize probability
    cur_prob_zx_norm = sum(cur_prob_zx);
    cur_prob_zx = cur_prob_zx / cur_prob_zx_norm;
    
    cur_prob_zy_norm = sum(cur_prob_zy);
    cur_prob_zy = cur_prob_zy / cur_prob_zy_norm;
  
    cur_prob_x_zx_norm = sum(cur_prob_x_zx+1e-10,1);
    cur_prob_x_zx = cur_prob_x_zx ./ repmat(cur_prob_x_zx_norm,nUser,1);
    
    cur_prob_y_zy_norm = sum(cur_prob_y_zy+1e-10,1);
    cur_prob_y_zy = cur_prob_y_zy ./ repmat(cur_prob_y_zy_norm,nItem,1);
  
    cur_prob_r_zxzy_norm = sum(cur_prob_r_zxzy+1e-10,3);
    cur_prob_r_zxzy = cur_prob_r_zxzy ./ repmat(cur_prob_r_zxzy_norm,[1,1,nRate]);
    
    
    %% measure change
    prob_zx_diff = cur_prob_zx - prob_zx;
    cg_prob_zx = norm(prob_zx_diff(:)) / norm(prob_zx(:));
    prob_zy_diff = cur_prob_zy - prob_zy;
    cg_prob_zy = norm(prob_zy_diff(:)) / norm(prob_zy(:));
    prob_x_zx_diff = cur_prob_x_zx - prob_x_zx;
    cg_prob_x_zx = norm(prob_x_zx_diff(:)) / norm(prob_x_zx(:));
    prob_y_zy_diff = cur_prob_y_zy - prob_y_zy;
    cg_prob_y_zy = norm(prob_y_zy_diff(:)) / norm(prob_y_zy(:));
    prob_r_zxzy_diff = cur_prob_r_zxzy - prob_r_zxzy;
    cg_prob_r_zxzy = norm(prob_r_zxzy_diff(:)) / norm(prob_r_zxzy(:));
    IterMeasure = max([cg_prob_zx cg_prob_zy cg_prob_x_zx cg_prob_y_zy cg_prob_r_zxzy]);
    
    fprintf('Training Iteration: %d, parameter change is: %f\n',IterCount,IterMeasure);
    
    prob_zx = cur_prob_zx;
    prob_zy = cur_prob_zy;
    prob_x_zx = cur_prob_x_zx;
    prob_y_zy = cur_prob_y_zy;
    prob_r_zxzy = cur_prob_r_zxzy;
end


model.prob_zx = prob_zx;
model.prob_zy = prob_zy;
model.prob_x_zx = prob_x_zx;
model.prob_y_zy = prob_y_zy;
model.prob_r_zxzy = prob_r_zxzy;
model.b = b;



