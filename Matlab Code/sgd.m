
close all
clear 

numFeature = 30;
learningRate = 0.001;
regulationRate = 0.001;

[userID,movieID,rating,time] = importfile('u1.base');
[userIDTest, movieIDTest, ratingTest, timeTest] = importfile('u1.test');

numUser = max(userID);
numMovie = max(movieID);

userFeature = rand(numUser, numFeature) * 0.1;
movieFeature = rand(numMovie, numFeature) *  0.1;

predRatings = userFeature * movieFeature';

training = 1;

loopCondition = true;


while loopCondition 

    RMSE(training) = 0;
    predRatings = userFeature * movieFeature';


for i=1:length(userID)
    
    tempMovieFeature = movieFeature(movieID(i),:);
    predictionForLoop = predRatings(userID(i), movieID(i));
    
    error = rating(i) - predictionForLoop; 
    
    movieFeature(movieID(i),:) = movieFeature(movieID(i),:) + learningRate* ( error * userFeature(userID(i),:)-regulationRate*movieFeature(movieID(i),:));
    userFeature(userID(i),:) = userFeature(userID(i),:) + learningRate * (error * tempMovieFeature - regulationRate*userFeature(userID(i),:));
    
    RMSE(training) = RMSE(training) + error^2;
    
end

RMSE(training) = RMSE(training)/length(userID);

testRMSE(training) = 0;

for i = 1:length(userIDTest)
    error = ratingTest(i) - predRatings(userIDTest(i),movieIDTest(i));
    
    testRMSE(training) = testRMSE(training) + error^2;
    
end

testRMSE(training) = testRMSE(training)/length(userIDTest);

disp(testRMSE(training))

plot(1:training,RMSE, 1:training,testRMSE); 
drawnow;

if training > 10 && testRMSE(training)/testRMSE(training-1) > 0.9999999
    loopCondition = false;
end

if training > 1000
    loopCondition = false;
end

training = training + 1;

end

userTSNE = tsne(userFeature, [], 2);
figure, scatter(userTSNE(:,1),userTSNE(:,2))

movieTSNE = tsne(movieFeature, [], 2);
figure, scatter(movieTSNE(:,1),movieTSNE(:,2))



    
    
    
    







