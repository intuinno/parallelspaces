% ------------------------------ Usage examples ------------------------------
m = 300;
n = 200;
k = 10;

A = rand(m,n);

% -------------------------------------------------------------------------
% Comment some of examples and try to execute!
% -------------------------------------------------------------------------

[W,H,iter,REC]=nmf(A,k,'tol',1e-3,'method','anls_bpp');

% for obtaining clustering index
[~, idx] = max(H);


%% 
% [W,H,iter,REC]=nmf(A,k,'verbose',2,'method','mu','max_iter',1000);
% [W,H,iter,REC]=nmf(A,k,'verbose',1,'method','hals','max_iter',1000);
% 
% algs = {'anls_bpp' 'anls_asgroup' 'anls_asgivens' 'anls_pgrad' 'anls_pqn' 'als' 'mu' 'hals'};
% for i=1:length(algs)
% 	[W,H,iter,REC]=nmf(A,k,'verbose',1,'method',algs{i});
% end
% 
% algs = {'anls_bpp' 'anls_asgroup' 'anls_asgivens' 'anls_pgrad' 'anls_pqn' 'als' 'mu' 'hals'};
% for i=1:length(algs)
% 	% Frobenius norm regularization test
% 	[W,H,iter,REC]=nmf(A,k,'tol',1e-3,'method',algs{i},'reg_w',[0.1 0],'reg_h',[0.8 0]);
% 	% L1-norm regularization test
% 	[W,H,iter,REC]=nmf(A,k,'tol',1e-3,'method',algs{i},'reg_w',[0.1 0],'reg_h',[0 0.8]);
% end
% 


%% nmf with missing value
% mask(i,j)= 1 if the value of A(m,n) is missing. 
mask = rand(m,n)<0.3;
[W,H]=cf_nmf(A,k,mask);

% for obtaining clustering index
[~, idx] = max(H);
