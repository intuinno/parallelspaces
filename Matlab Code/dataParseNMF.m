tic 
close all

clear



load 1Kratings


[numUser numMovie] = size(ratings);

ratingsNor = ratings/5.0;

mask = ratings == 0;

[W H] = cf_nmf(ratingsNor,20, mask)



% csvwrite('movieSpace.csv',movieSpace);
% csvwrite('userSpace.csv',userSpace);
% csvwrite('ratings.csv',ratings);



toc
