tic 
close all

clear

if matlabpool('size') == 0 
	
%	matlabpool 

end

load 1Kratings

colormap('bone');

[numUser numMovie] = size(ratings);

movieRatings = zeros(numMovie ,2);
userRatings = zeros (numUser,2);

count = 0;

TempRating = 0;

for i =1:numMovie 
	
	for j = 1:numUser 

		if ratings(j,i) > 0 
			
			count = count+1;
			
			TempRating = TempRating + ratings(j,i);
		end
		
	end
	
	
	movieRatings(i,1) = count;
	
	movieRatings(i,2) = TempRating/count;
	
	count =0;
	TempRating =0;
end

for i =1:numUser 
	
	for j = 1:numMovie 

		if ratings(i,j) > 0 
			
			count = count+1;
			
			TempRating = TempRating + ratings(i,j);
		end
		
	end
	
	
	userRatings(i,1) = count;
	
	userRatings(i,2) = TempRating/count;
	
	count =0;
	TempRating =0;
	
	
end


		
[wcoeff,score,latent,tsquared] = princomp(ratings');
%figure('Color', [0 0 0]);

subplot(1,2,1); 
scatter(score(:,1),score(:,2),movieRatings(:,1)./4,movieRatings(:,2),'filled','HitTest','On','HitTestArea','On')
xlabel('1st Principal Component')
ylabel('2nd Principal Component')
set(gca,'Color',[0,0,0]);
title('Movie Space','Color','k')
colormap('bone')
caxis([0 5])

score(:,1) = scaledown(score(:,1));
score(:,2) = scaledown(score(:,2));

movieSpace = [score(:,1), score(:,2), movieRatings(:,1), (movieRatings(:,2))];


[wcoeff,score,latent,tsquared] = princomp(ratings);
subplot(1,2,2); 
scatter(score(:,1),score(:,2),userRatings(:,1)./4,userRatings(:,2),'filled','HitTest','On','HitTestArea','On')
xlabel('1st Principal Component')
ylabel('2nd Principal Component')
set(gca,'Color',[0,0,0]);
title('User Space','Color','k')
colormap('bone')
caxis([0 5])


score(:,1) = scaledown(score(:,1));
score(:,2) = scaledown(score(:,2));

userSpace = [score(:,1), score(:,2), (userRatings(:,1)), (userRatings(:,2))];

%json = savejson('movieSpace',movieSpace,'movieSpace.json');
%json = savejson('userSpace',userSpace,'userSpace.json');

csvwrite('movieSpace.csv',movieSpace);
csvwrite('userSpace.csv',userSpace);
csvwrite('ratings.csv',ratings);



toc
