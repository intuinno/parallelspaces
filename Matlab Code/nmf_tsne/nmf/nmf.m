% Nonnegative Matrix Factorization : Algorithms Toolbox
%
% Written by Jingu Kim (jingu.kim@gmail.com)
%            School of Computational Science and Engineering,
%            Georgia Institute of Technology
%
% Please send bug reports, comments, or questions to Jingu Kim.
% This code comes with no guarantee or warranty of any kind.
%
% Reference:
%		 Jingu Kim and Haesun Park,
%		 Fast Nonnegative Matrix Factorization: An Active-set-like Method And Comparisons,
%		 SIAM Journal on Scientific Computing (SISC), 33(6), pp. 3261-3281, 2011.
%
% Last modified on 02/22/2012
%
% <Inputs>
%        A : Input data matrix (m x n)
%        k : Target low-rank
%
%        (Below are optional arguments: can be set by providing name-value pairs)
%
%        METHOD : Algorithm for solving NMF. One of the following values:
%				  'anls_bpp' 'hals' 'anls_asgroup' 'anls_asgivens' 'anls_pgrad' 'anls_pqn' 'als' 'mu'
%				  See above paper (and references therein) for the details of these algorithms.
%				  Default is 'anls_bpp'.
%        TOL : Stopping tolerance. Default is 1e-3. 
%              If you want to obtain a more accurate solution, decrease TOL and increase MAX_ITER at the same time.
%        MAX_ITER : Maximum number of iterations. Default is 500.
%        MIN_ITER : Minimum number of iterations. Default is 20.
%        MAX_TIME : Maximum amount of time in seconds. Default is 1,000,000.
%		 INIT : A struct containing initial values. INIT.W and INIT.H should contain initial values of
%               W and H of size (m x k) and (k x n), respectively.
%        VERBOSE : 0 (default) - No debugging information is collected.
%                  1 (debugging/experimental purpose) - History of computation is returned. See 'REC' variable.
%                  2 (debugging/experimental purpose) - History of computation is additionally printed on screen.
%		 REG_W, REG_H : Regularization parameters for W and H.
%                       Both REG_W and REG_H should be vector of two nonnegative numbers.
%                       The first component is a parameter with Frobenius norm regularization, and
%                       the second component is a parameter with L1-norm regularization.
%                       For example, to promote sparsity in H, one might set REG_W = [alpha 0] and REG_H = [0 beta]
%                       where alpha and beta are positive numbers. See above paper for more details.
%                       Defaut is [0 0] for both REG_W and REG_H, which means no regularization.
% <Outputs>
%        W : Obtained basis matrix (m x k)
%        H : Obtained coefficient matrix (k x n)
%        iter : Number of iterations
%        HIS : (debugging/experimental purpose) Auxiliary information about the execution
% <Usage Examples>
%        nmf(A,10)
%        nmf(A,20,'verbose',2)
%        nmf(A,20,'verbose',1,'method','anls_bpp')
%        nmf(A,20,'verbose',1,'method','hals')
%        nmf(A,20,'verbose',1,'reg_w',[0.1 0],'reg_h',[0 0.5])

function [W,H,iter,REC]=nmf(A,k,varargin)
	% parse parameters
	params = inputParser;
	params.addParamValue('method'        ,'anls_bpp',@(x) ischar(x) );
%	params.addParamValue('method'        ,'hals',@(x) ischar(x) );
	params.addParamValue('tol'           ,1e-2      ,@(x) isscalar(x) & x > 0);
	params.addParamValue('min_iter'      ,20        ,@(x) isscalar(x) & x > 0);
	params.addParamValue('max_iter'      ,100      ,@(x) isscalar(x) & x > 0);
	params.addParamValue('max_time'      ,1e6       ,@(x) isscalar(x) & x > 0);
	params.addParamValue('init'          ,struct([]),@(x) isstruct(x));
	params.addParamValue('verbose'       ,1         ,@(x) isscalar(x) & x >= 0);
	params.addParamValue('reg_w'         ,[0 0]     ,@(x) isvector(x) & length(x) == 2);
	params.addParamValue('reg_h'         ,[0 0]     ,@(x) isvector(x) & length(x) == 2);
	% The following options are reserved for debugging/experimental purposes. 
	% Make sure to understand them before making changes
	params.addParamValue('subparams'     ,struct([]),@(x) isstruct(x) );
	params.addParamValue('track_grad'    ,1         ,@(x) isscalar(x) & x >= 0);
	params.addParamValue('track_prev'    ,1         ,@(x) isscalar(x) & x >= 0);
	params.addParamValue('stop_criterion',2         ,@(x) isscalar(x) & x >= 0);
	params.parse(varargin{:});

%     % joyfull
%     save tmp_data;

    if size(A,2)==3
        A=full(sparse(A(:,1),A(:,2),A(:,3)));
        size(A)
    %     X=X(:,data_ind);
    %     X=X((sum(abs(X),2)~=0)',:);
    %     size(X)
    end
    % X=X-repmat(mean(X,2),1,size(X,2));
%     params

	% copy from params object
	[m,n] = size(A);
	par = params.Results;
	par.m = m;
	par.n = n;
	par.k = k;

	% Stopping criteria are based on the gradient information.
	% Hence, 'track_grad' option needs to be turned on to use a criterion.
	if par.stop_criterion > 0
		par.track_grad = 1;
	end

	% initialize
	if isempty(par.init)
    	W = rand(m,k); H = rand(k,n);
%        W = WW; H = HH;
	else
		W = par.init.W; H = par.init.H;
	end

	% This variable is for analysis/debugging, so it does not affect the output (W,H) of this program
    REC = struct([]);

    if par.verbose          % Collect initial information for analysis/debugging
		clear('init');
		init.norm_A      = norm(A,'fro'); 
		init.norm_W		 = norm(W,'fro');
		init.norm_H		 = norm(H,'fro');
		init.baseObj	 = getObj((init.norm_A)^2,W,H,par);
		if par.track_grad
        	[gradW,gradH] 	 = getGradient(A,W,H,par);
			init.normGr_W    = norm(gradW,'fro');
			init.normGr_H    = norm(gradH,'fro');
			init.SC_NM_PGRAD = getInitCriterion(1,A,W,H,par,gradW,gradH);
			init.SC_PGRAD    = getInitCriterion(2,A,W,H,par,gradW,gradH);
			init.SC_DELTA    = getInitCriterion(3,A,W,H,par,gradW,gradH);
		else
        	gradW = 0; gradH = 0;
		end

		if par.track_prev 
			prev_W = W; prev_H = H;
		else
			prev_W = 0; prev_H = 0;
		end
			
		ver = prepareHIS(A,W,H,prev_W,prev_H,init,par,0,0,gradW,gradH);
		REC(1).init = init;
		REC.HIS = ver;

        if par.verbose == 2
			display(init);
		end

        tPrev = cputime;
    end

	initializer= str2func([par.method,'_initializer']);
	iterSolver = str2func([par.method,'_iterSolver']);
	iterLogger = str2func([par.method,'_iterLogger']);
	[W,H,par,val,ver] = feval(initializer,A,W,H,par);

	if par.verbose & ~isempty(ver)
		tTemp = cputime;
		REC.HIS = saveHIS(1,ver,REC.HIS);
		tPrev = tPrev+(cputime-tTemp);
	end

	REC(1).par = par;
	REC.start_time = datestr(now);
    display(par);

    tStart = cputime;, tTotal = 0;
	if par.track_grad
    	initSC = getInitCriterion(par.stop_criterion,A,W,H,par);
	end
    SCconv = 0; SC_COUNT = 3;

%    obj = zeros(par.max_iter,1);
    for iter=1:par.max_iter

		% Actual work of this iteration is executed here.
		[W,H,gradW,gradH,val] = feval(iterSolver,A,W,H,iter,par,val);
        iter
%        [iter norm(A - W*H, 'fro')]
%        obj(iter) = norm(A - W*H,'fro');

        if par.verbose          % Collect information for analysis/debugging
            elapsed = cputime-tPrev;
            tTotal = tTotal + elapsed;

            clear('ver');
			ver = prepareHIS(A,W,H,prev_W,prev_H,init,par,iter,elapsed,gradW,gradH);

			ver = feval(iterLogger,ver,par,val,W,H,prev_W,prev_H);
			REC.HIS = saveHIS(iter+1,ver,REC.HIS);

			if par.track_prev, prev_W = W; prev_H = H; end
            if par.verbose == 2, display(ver);, end
            tPrev = cputime;
        end

        if (par.verbose && (tTotal > par.max_time)) || (~par.verbose && ((cputime-tStart)>par.max_time))
            break;
        elseif par.track_grad
            SC = getStopCriterion(par.stop_criterion,A,W,H,par,gradW,gradH);
            if (SC/initSC <= par.tol)
                SCconv = SCconv + 1;
%                if (SCconv >= SC_COUNT), break;, end
            else
                SCconv = 0;
            end
        end
    end
    [m,n]=size(A);
	[W,H]=normalize_by_W(W,H);
    
    if par.verbose
		final.elapsed_total = sum(REC.HIS.elapsed);
    else
        final.elapsed_total = cputime-tStart;
    end
    final.iterations     = iter;
% 	sqErr = getSquaredError(A,W,H,init);
% 	final.relative_error = sqrt(sqErr)/init.norm_A;
% 	final.relative_obj	 = getObj(sqErr,W,H,par)/init.baseObj;
    final.W_density      = length(find(W>0))/(m*k);
    final.H_density      = length(find(H>0))/(n*k);

    if par.verbose
		REC.final = final;
	end
	REC.finish_time = datestr(now);
    display(final); 
end

%----------------------------------------------------------------------------
%                              Implementation of methods
%----------------------------------------------------------------------------

%----------------- ANLS with Block Principal Pivoting Method --------------------------

function [W,H,par,val,ver] = anls_bpp_initializer(A,W,H,par)
	H = zeros(size(H));

	ver.turnZr_W  = 0;
	ver.turnZr_H  = 0;
	ver.turnNz_W  = 0;
	ver.turnNz_H  = 0;
	ver.numChol_W = 0;
	ver.numChol_H = 0;
	ver.numEq_W   = 0;
	ver.numEq_H   = 0;
	ver.suc_W     = 0;
	ver.suc_H     = 0;

	val(1).WtA = W'*A;
	val.WtW = W'*W;
end

function [W,H,gradW,gradH,val] = anls_bpp_iterSolver(A,W,H,iter,par,val)

	WtW_reg = applyReg(val.WtW,par,par.reg_h);
    [H,temp,suc_H,numChol_H,numEq_H] = nnlsm_blockpivot(WtW_reg,val.WtA,1,H);

	HHt_reg = applyReg(H*H',par,par.reg_w);
    [W,gradW,suc_W,numChol_W,numEq_W] = nnlsm_blockpivot(HHt_reg,H*A',1,W');
	W = W';

	val.WtA = W'*A;
	val.WtW = W'*W;

	if par.track_grad
		gradW = gradW';
		gradH = getGradientOne(val.WtW,val.WtA,H,par.reg_h,par);
	else
		gradW = 0;gradH =0;
	end

	val(1).numChol_W = numChol_W;
	val.numChol_H = numChol_H;
	val.numEq_W = numEq_W;
	val.numEq_H = numEq_H;
	val.suc_W = suc_W;
	val.suc_H = suc_H;
end

function [ver] = anls_bpp_iterLogger(ver,par,val,W,H,prev_W,prev_H)
	if par.track_prev
		ver.turnZr_W	= length(find( (prev_W>0) & (W==0) ))/(par.m*par.k);
		ver.turnZr_H	= length(find( (prev_H>0) & (H==0) ))/(par.n*par.k);
		ver.turnNz_W	= length(find( (prev_W==0) & (W>0) ))/(par.m*par.k);
		ver.turnNz_H	= length(find( (prev_H==0) & (H>0) ))/(par.n*par.k);
	end
	ver.numChol_W   = val.numChol_W;
	ver.numChol_H   = val.numChol_H;
	ver.numEq_W     = val.numEq_W;
	ver.numEq_H     = val.numEq_H;
	ver.suc_W       = val.suc_W;
	ver.suc_H       = val.suc_H;
end

%----------------- ANLS with Active Set Method / Givens Updating --------------------------

function [W,H,par,val,ver] = anls_asgivens_initializer(A,W,H,par)
	H = zeros(size(H));

	ver.turnZr_W  = 0;
	ver.turnZr_H  = 0;
	ver.turnNz_W  = 0;
	ver.turnNz_H  = 0;
	ver.numChol_W = 0;
	ver.numChol_H = 0;
	ver.suc_W     = 0;
	ver.suc_H     = 0;

	val(1).WtA = W'*A;
	val.WtW = W'*W;
end

function [W,H,gradW,gradH,val] = anls_asgivens_iterSolver(A,W,H,iter,par,val)
	WtW_reg = applyReg(val.WtW,par,par.reg_h);
	ow = 0;
	suc_H = zeros(1,size(H,2));
	numChol_H = zeros(1,size(H,2));
	for i=1:size(H,2)
    	[H(:,i),temp,suc_H(i),numChol_H(i)] = nnls1_asgivens(WtW_reg,val.WtA(:,i),ow,1,H(:,i));
	end

	suc_W = zeros(1,size(W,1));
	numChol_W = zeros(1,size(W,1));

	HHt_reg = applyReg(H*H',par,par.reg_w);
	HAt = H*A';
	Wt = W';
	gradWt = zeros(size(Wt));
	for i=1:size(W,1)
    	[Wt(:,i),gradWt(:,i),suc_W(i),numChol_W(i)] = nnls1_asgivens(HHt_reg,HAt(:,i),ow,1,Wt(:,i));
	end
	W = Wt';

	val.WtA = W'*A;
	val.WtW = W'*W;

	if par.track_grad
		gradW = gradWt'; 
		gradH = getGradientOne(val.WtW,val.WtA,H,par.reg_h,par);
	else
		gradW = 0; gradH =0;
	end

	val(1).numChol_W = sum(numChol_W);
	val.numChol_H = sum(numChol_H);
	val.suc_W = any(suc_W);
	val.suc_H = any(suc_H);
end

function [ver] = anls_asgivens_iterLogger(ver,par,val,W,H,prev_W,prev_H)
	if par.track_prev
		ver.turnZr_W	= length(find( (prev_W>0) & (W==0) ))/(par.m*par.k);
		ver.turnZr_H	= length(find( (prev_H>0) & (H==0) ))/(par.n*par.k);
		ver.turnNz_W	= length(find( (prev_W==0) & (W>0) ))/(par.m*par.k);
		ver.turnNz_H	= length(find( (prev_H==0) & (H>0) ))/(par.n*par.k);
	end
	ver.numChol_W   = val.numChol_W;
	ver.numChol_H   = val.numChol_H;
	ver.suc_W       = val.suc_W;
	ver.suc_H       = val.suc_H;
end

%----------- ANLS with Active Set Method / Column Grouping / No Overwrite -----------------

function [W,H,par,val,ver] = anls_asgroup_initializer(A,W,H,par)
	[W,H,par,val,ver] = anls_bpp_initializer(A,W,H,par);
end

function [W,H,gradW,gradH,val] = anls_asgroup_iterSolver(A,W,H,iter,par,val)
	WtW_reg = applyReg(val.WtW,par,par.reg_h);
	ow = 0;
    [H,temp,suc_H,numChol_H,numEq_H] = nnlsm_activeset(WtW_reg,val.WtA,ow,1,H);

	HHt_reg = applyReg(H*H',par,par.reg_w);
    [W,gradW,suc_W,numChol_W,numEq_W] = nnlsm_activeset(HHt_reg,H*A',ow,1,W');
	W = W';

	val.WtA = W'*A;
	val.WtW = W'*W;

	if par.track_grad
		gradW = gradW'; 
		gradH = getGradientOne(val.WtW,val.WtA,H,par.reg_h,par);
	else
		gradW = 0; gradH =0;
	end

	val(1).numChol_W = numChol_W;
	val.numChol_H = numChol_H;
	val.numEq_W = numEq_W;
	val.numEq_H = numEq_H;
	val.suc_W = suc_W;
	val.suc_H = suc_H;
end

function [ver] = anls_asgroup_iterLogger(ver,par,val,W,H,prev_W,prev_H)
	ver = anls_bpp_iterLogger(ver,par,val,W,H,prev_W,prev_H);
end

%----------------- ANLS with Projected Gradient Method --------------------------

function [W,H,par,val,ver] = anls_pgrad_initializer(A,W,H,par)
	if isempty(par.subparams)
		par.subparams(1).subtol_init   = 0.1;
		par.subparams.reduce_factor = 0.1;
		par.subparams.num_warmup_iter = 5;
		par.subparams.min_subiter = 1;
		par.subparams.max_subiter = 100;
		par.subparams.min_tol = 1e-10;
	end

    [gradW,gradH] = getGradient(A,W,H,par);
    val(1).tol_W = par.subparams.subtol_init*norm(gradW,'fro');
    val.tol_H = par.subparams.subtol_init*norm(gradH,'fro');

	ver.subIter_W 		= 0;
	ver.subIter_H 		= 0;
	ver.numLineSearch_W = 0;
	ver.numLineSearch_H = 0;
end

function [W,H,gradW,gradH,val] = anls_pgrad_iterSolver(A,W,H,iter,par,val)

	if (iter>1 && iter<=par.subparams.num_warmup_iter)
		val.tol_W = val.tol_W * par.subparams.reduce_factor;
		val.tol_H = val.tol_H * par.subparams.reduce_factor;
	end

	WtW_reg = applyReg(W'*W,par,par.reg_h);
	WtA = W'*A;
    [H,gradHX,subIter_H,numLineSearch_H] = nnlssub_projgrad(WtW_reg,WtA,val.tol_H,H,par.subparams.max_subiter,1);
    if (iter > par.subparams.num_warmup_iter) && (subIter_H < par.subparams.min_subiter) && (val.tol_H >= par.subparams.min_tol)
        val.tol_H = par.subparams.reduce_factor * val.tol_H;
        [H,gradHX,subAddH,lsAddH] = nnlssub_projgrad(WtW_reg,WtA,val.tol_H,H,par.subparams.max_subiter,1);
        subIter_H = subIter_H + subAddH;
        numLineSearch_H = numLineSearch_H + lsAddH;
    end

	HHt_reg = applyReg(H*H',par,par.reg_w);
	HAt = H*A';
    [W,gradW,subIter_W,numLineSearch_W] = nnlssub_projgrad(HHt_reg,HAt,val.tol_W,W',par.subparams.max_subiter,1);
    if (iter > par.subparams.num_warmup_iter) && (subIter_W < par.subparams.min_subiter) && (val.tol_W >= par.subparams.min_tol)
        val.tol_W = par.subparams.reduce_factor * val.tol_W;
        [W,gradW,subAddW,lsAddW] = nnlssub_projgrad(HHt_reg,HAt,val.tol_W,W,par.subparams.max_subiter,1);
        subIter_W = subIter_W + subAddW;
        numLineSearch_W = numLineSearch_W + lsAddW;
    end
	W = W';
	if par.track_grad
		gradW = gradW';
		gradH = getGradientOne(W'*W,W'*A,H,par.reg_h,par);
	else
		gradH = 0; gradW = 0;
	end

	val(1).subIter_W 	= subIter_W;
	val.subIter_H 		= subIter_H;
	val.numLineSearch_W = numLineSearch_W;
	val.numLineSearch_H = numLineSearch_H;
end

function [ver] = anls_pgrad_iterLogger(ver,par,val,W,H,prev_W,prev_H)
	ver.tol_W			= val.tol_W;
	ver.tol_H			= val.tol_H;
	ver.subIter_W 		= val.subIter_W;
	ver.subIter_H 		= val.subIter_H;
	ver.numLineSearch_W = val.numLineSearch_W;
	ver.numLineSearch_H = val.numLineSearch_H;
end

%----------------- ANLS with Projected Quasi Newton Method --------------------------

function [W,H,par,val,ver] = anls_pqn_initializer(A,W,H,par)
	[W,H,par,val,ver] = anls_pgrad_initializer(A,W,H,par);
end

function [W,H,gradW,gradH,val] = anls_pqn_iterSolver(A,W,H,iter,par,val)

	if (iter>1 && iter<=par.subparams.num_warmup_iter)
		val.tol_W = val.tol_W * par.subparams.reduce_factor;
		val.tol_H = val.tol_H * par.subparams.reduce_factor;
	end

	WtW_reg = applyReg(W'*W,par,par.reg_h);
	WtA = W'*A;
    [H,gradHX,subIter_H] = nnlssub_projnewton_mod(WtW_reg,WtA,val.tol_H,H,par.subparams.max_subiter,1);
    if (iter > par.subparams.num_warmup_iter) && (subIter_H/par.n < par.subparams.min_subiter) && (val.tol_H >= par.subparams.min_tol)
        val.tol_H = par.subparams.reduce_factor * val.tol_H;
        [H,gradHX,subAddH] = nnlssub_projnewton_mod(WtW_reg,WtA,val.tol_H,H,par.subparams.max_subiter,1);
        subIter_H = subIter_H + subAddH;
    end

	HHt_reg = applyReg(H*H',par,par.reg_w);
	HAt = H*A';
    [W,gradW,subIter_W] = nnlssub_projnewton_mod(HHt_reg,HAt,val.tol_W,W',par.subparams.max_subiter,1);
    if (iter > par.subparams.num_warmup_iter) && (subIter_W/par.m < par.subparams.min_subiter) && (val.tol_W >= par.subparams.min_tol)
        val.tol_W = par.subparams.reduce_factor * val.tol_W;
        [W,gradW,subAddW] = nnlssub_projnewton_mod(HHt_reg,HAt,val.tol_W,W,par.subparams.max_subiter,1);
        subIter_W = subIter_W + subAddW;
    end
	W = W';

	if par.track_grad
		gradW = gradW';
		gradH = getGradientOne(W'*W,W'*A,H,par.reg_h,par);
	else
		gradH = 0; gradW = 0;
	end

	val(1).subIter_W = subIter_W;
	val.subIter_H = subIter_H;
end

function [ver] = anls_pqn_iterLogger(ver,par,val,W,H,prev_W,prev_H)
	ver.tol_W			= val.tol_W;
	ver.tol_H			= val.tol_H;
	ver.subIter_W 		= val.subIter_W;
	ver.subIter_H 		= val.subIter_H;
end

%----------------- Alternating Least Squares Method --------------------------

function [W,H,par,val,ver] = als_initializer(A,W,H,par)
	ver = struct([]);

	val.WtA = W'*A;
	val.WtW = W'*W;
end

function [W,H,gradW,gradH,val] = als_iterSolver(A,W,H,iter,par,val)
	WtW_reg = applyReg(val.WtW,par,par.reg_h);
    H = WtW_reg\val.WtA;
    H(H<0)=0;

    AHt = A*H';
	HHt_reg = applyReg(H*H',par,par.reg_w);
    Wt = HHt_reg\AHt'; W=Wt';
    W(W<0)=0;

    % normalize : necessary for ALS
	[W,H,weights] = normalize_by_W(W,H);
	D = diag(weights);

	val.WtA = W'*A;
	val.WtW = W'*W;
    AHt = AHt*D;
    HHt_reg = D*HHt_reg*D;

	if par.track_grad
    	gradW = W*HHt_reg - AHt;
		gradH = getGradientOne(val.WtW,val.WtA,H,par.reg_h,par);
	else
		gradH = 0; gradW = 0;
	end
end

function [ver] = als_iterLogger(ver,par,val,W,H,prev_W,prev_H)
end

%----------------- Multiplicatve Updating Method --------------------------

function [W,H,par,val,ver] = mu_initializer(A,W,H,par)
	ver = struct([]);

	val.WtA = W'*A;
	val.WtW = W'*W;
end

function [W,H,gradW,gradH,val] = mu_iterSolver(A,W,H,iter,par,val)
	epsilon = 1e-16;

	WtW_reg = applyReg(val.WtW,par,par.reg_h);
    H = H.*val.WtA./(WtW_reg*H + epsilon);
%    WtW_reg = W'*W;
%    H = H.*(W'*A)./(WtW_reg*H + epsilon);

	HHt_reg = applyReg(H*H',par,par.reg_w);
	AHt = A*H';
    W = W.*AHt./(W*HHt_reg + epsilon);
%    HHt_reg = H*H';
%    AHt = A*H';
%    W = W.*AHt./(W*HHt_reg+epsilon);

	val.WtA = W'*A;
	val.WtW = W'*W;

	if par.track_grad
    	gradW = W*HHt_reg - AHt;
		gradH = getGradientOne(val.WtW,val.WtA,H,par.reg_h,par);
	else
		gradH = 0; gradW = 0;
    end
    norm(A-W*H,'fro')
end

function [ver] = mu_iterLogger(ver,par,val,W,H,prev_W,prev_H)
end

%----------------- HALS Method : Algorith 2 of Cichocki and Phan -----------------------

function [W,H,par,val,ver] = hals_initializer(A,W,H,par)
	[W,H]=normalize_by_W(W,H);

	val = struct([]);
	ver = struct([]);
end

function [W,H,gradW,gradH,val] = hals_iterSolver(A,W,H,iter,par,val)
	epsilon = 1e-16;

	WtA = W'*A;
	WtW = W'*W;
	WtW_reg = applyReg(WtW,par,par.reg_h);
	for i = 1:par.k
		H(i,:) = max(H(i,:) + WtA(i,:) - WtW_reg(i,:) * H,epsilon);
    end

    AHt = A*H';
	HHt_reg = applyReg(H*H',par,par.reg_w);
	for i = 1:par.k
		W(:,i) = max(W(:,i) * HHt_reg(i,i) + AHt(:,i) - W * HHt_reg(:,i),epsilon);
		if sum(W(:,i))>0
			W(:,i) = W(:,i)/norm(W(:,i));
		end
	end

	if par.track_grad
    	gradW = W*HHt_reg - AHt;
		gradH = getGradientOne(W'*W,W'*A,H,par.reg_h,par);
	else
		gradH = 0; gradW = 0;
	end
end

function [ver] = hals_iterLogger(ver,par,val,W,H,prev_W,prev_H)
end

%----------------------------------------------------------------------------------------------
%                                    Utility Functions 
%----------------------------------------------------------------------------------------------

% This function prepares information about execution for a experiment purpose
function ver = prepareHIS(A,W,H,prev_W,prev_H,init,par,iter,elapsed,gradW,gradH)
	ver.iter          = iter;
	ver.elapsed       = elapsed;

	sqErr = getSquaredError(A,W,H,init);
	ver.rel_Error 		= sqrt(sqErr)/init.norm_A;
	ver.rel_Obj			= getObj(sqErr,W,H,par)/init.baseObj;
	ver.norm_W		  = norm(W,'fro');
	ver.norm_H		  = norm(H,'fro');
	if par.track_prev
		ver.rel_Change_W  = norm(W-prev_W,'fro')/init.norm_W;
		ver.rel_Change_H  = norm(H-prev_H,'fro')/init.norm_H;
	end
	if par.track_grad
		ver.rel_NrPGrad_W = norm(projGradient(W,gradW),'fro')/init.normGr_W;
		ver.rel_NrPGrad_H = norm(projGradient(H,gradH),'fro')/init.normGr_H;
		ver.SC_NM_PGRAD   = getStopCriterion(1,A,W,H,par,gradW,gradH)/init.SC_NM_PGRAD;
		ver.SC_PGRAD      = getStopCriterion(2,A,W,H,par,gradW,gradH)/init.SC_PGRAD;
		ver.SC_DELTA      = getStopCriterion(3,A,W,H,par,gradW,gradH)/init.SC_DELTA; 
	end
	ver.density_W     = length(find(W>0))/(par.m*par.k);
	ver.density_H     = length(find(H>0))/(par.n*par.k);
end

% Execution information is collected in HIS variable
function HIS = saveHIS(idx,ver,HIS)
	%idx = length(HIS.iter)+1;
	fldnames = fieldnames(ver);

	for i=1:length(fldnames)
		flname = fldnames{i};
		HIS.(flname)(idx) = ver.(flname);
	end
end

%-------------------------------------------------------------------------------
function retVal = getInitCriterion(stopRule,A,W,H,par,gradW,gradH)
% STOPPING_RULE : 1 - Normalized proj. gradient
%                 2 - Proj. gradient
%                 3 - Delta by H. Kim
%                 0 - None (want to stop by MAX_ITER or MAX_TIME)
    if nargin~=7
        [gradW,gradH] = getGradient(A,W,H,par);
    end
    [m,k]=size(W);, [k,n]=size(H);, numAll=(m*k)+(k*n);
    switch stopRule
        case 1
            retVal = norm([gradW(:); gradH(:)])/numAll;
        case 2
            retVal = norm([gradW(:); gradH(:)]);
        case 3
            retVal = getStopCriterion(3,A,W,H,par,gradW,gradH);
        case 0
            retVal = 1;
    end
end
%-------------------------------------------------------------------------------
function retVal = getStopCriterion(stopRule,A,W,H,par,gradW,gradH)
% STOPPING_RULE : 1 - Normalized proj. gradient
%                 2 - Proj. gradient
%                 3 - Delta by H. Kim
%                 0 - None (want to stop by MAX_ITER or MAX_TIME)
    if nargin~=7
        [gradW,gradH] = getGradient(A,W,H,par);
    end

    switch stopRule
        case 1
            pGradW = projGradient(W,gradW);
            pGradH = projGradient(H,gradH);
            pGrad = [pGradW(:); pGradH(:)];
            retVal = norm(pGrad)/length(pGrad);
        case 2
            pGradW = projGradient(W,gradW);
            pGradH = projGradient(H,gradH);
            pGrad = [pGradW(:); pGradH(:)];
            retVal = norm(pGrad);
        case 3
            resmat=min(H,gradH); resvec=resmat(:);
            resmat=min(W,gradW); resvec=[resvec; resmat(:)]; 
            deltao=norm(resvec,1); %L1-norm
            num_notconv=length(find(abs(resvec)>0));
            retVal=deltao/num_notconv;
        case 0
            retVal = 1e100;
    end
end
%-------------------------------------------------------------------------------
function sqErr = getSquaredError(A,W,H,init)
	sqErr = max((init.norm_A)^2 - 2*trace(H*(A'*W))+trace((W'*W)*(H*H')),0 );
end

function retVal = getObj(sqErr,W,H,par)
	retVal = 0.5 * sqErr;
	retVal = retVal + par.reg_w(1) * sum(sum(W.*W));
	retVal = retVal + par.reg_w(2) * sum(sum(W,2).^2);
	retVal = retVal + par.reg_h(1) * sum(sum(H.*H));
	retVal = retVal + par.reg_h(2) * sum(sum(H,1).^2);
end

function AtA = applyReg(AtA,par,reg)
	% Frobenius norm regularization
	if reg(1) > 0
		AtA = AtA + 2 * reg(1) * eye(par.k);
	end
	% L1-norm regularization
	if reg(2) > 0
		AtA = AtA + 2 * reg(2) * ones(par.k,par.k);
	end
end

function [grad] = modifyGradient(grad,X,reg,par)
	if reg(1) > 0
		grad = grad + 2 * reg(1) * X;
	end
	if reg(2) > 0
		grad = grad + 2 * reg(2) * ones(par.k,par.k) * X;
	end
end

function [grad] = getGradientOne(AtA,AtB,X,reg,par)
	grad = AtA*X - AtB;
	grad = modifyGradient(grad,X,reg,par);
end

function [gradW,gradH] = getGradient(A,W,H,par)
	HHt = H*H';
	HHt_reg = applyReg(HHt,par,par.reg_w);

	WtW = W'*W;
	WtW_reg = applyReg(WtW,par,par.reg_h);

    gradW = W*HHt_reg - A*H';
    gradH = WtW_reg*H - W'*A;
end

%-------------------------------------------------------------------------------
function pGradF = projGradient(F,gradF)
	pGradF = gradF(gradF<0|F>0);
end

%-------------------------------------------------------------------------------
function [W,H,weights] = normalize_by_W(W,H)
    norm2=sqrt(sum(W.^2,1));
    toNormalize = norm2>0;

    W(:,toNormalize) = W(:,toNormalize)./repmat(norm2(toNormalize),size(W,1),1);
    H(toNormalize,:) = H(toNormalize,:).*repmat(norm2(toNormalize)',1,size(H,2));

	weights = ones(size(norm2));
	weights(toNormalize) = norm2(toNormalize);
end
