function [W,H]=cf_nmf(A,k,mask,varargin)
	% copy from params object
	[m,n] = size(A);

	% initialize
 	W = rand(m,k); H = rand(k,n);

    max_iter = 1000;

    save_time = zeros(2,1);
	epsilon = 1e-16;
    maskA = A;
    maskA(mask) = 0;
    

    maskW = zeros(m,n);
    maskH = zeros(m,n);
    for iter=1:max_iter
        tic;
        WtA = W'*maskA;
        WtW = W'*W;

        maskW = maskA;
        for i = 1:k
            maskW = bsxfun(@times, W(:,i), maskW);
%             maskW = repmat(W(:,i), [1 n]) .* maskA;
            normW = sqrt(sum(maskW.^2,1));
%            H(i,:) = H(i,:) ./ diag(maskW);
            H(i,:) = max(H(i,:) + (WtA(i,:) - WtW(i,:) * H)./normW,epsilon);
        end
        save_time(1) = toc;

        tic;
        AHt = maskA*H';
        HHt = H*H';
        
        maskH = maskA;
        for i = 1:k
            maskH = bsxfun(@times, H(i,:), maskH);
%            maskH = repmat(H(i,:), [m 1]) .* maskA;
            normH = sqrt(sum(maskH.^2,2));
%            W(:,i) = W(:,i) ./ diag(maskH);
            W(:,i) = max(W(:,i) + (AHt(:,i) - W*HHt(:,i))./normH, epsilon);
%             W(:,i) = max(W(:,i) * HHt(i,i) + AHt(:,i) - W * HHt(:,i),epsilon);
%             if sum(W(:,i))>0
%                 W(:,i) = W(:,i)/norm(W(:,i));
%             end
        end
        save_time(2) = toc;
        if mod(iter,100)==0
%            [iter save_time(1) save_time(2)]
            iter
        end
    end
end
