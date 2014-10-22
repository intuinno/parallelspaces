load mnist_test

idx = randsample(size(test_X,1),200);

X = test_X(idx,:);
labels = test_labels(idx);
no_dims = 2

figure(1);clf;
ydata = tsne(X, labels, no_dims);

shrink_factor = .5;

figure(2);clf;
ydata = tsne_sup(X, labels, shrink_factor, no_dims);
