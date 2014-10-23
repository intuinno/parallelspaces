
close all
clear

numFeature = 100;
learningRate = 0.001;
regulationRate = 0.001;

[userID,movieID,rating,time] = importfile('u.data');

numUser = max(userID);
numMovie = max(movieID);

globalAverage = mean(rating);

% Get user bias

for i=1:numUser
    biasUser(i) = mean(rating(userID == i)) - globalAverage;
end

% Get movie bias

for i=1:numMovie
    if length(rating(movieID == i)) ~= 0
        biasMovie(i) = mean(rating(movieID == i)) - globalAverage;
    else
        biasMovie(i) = 0;
    end
    
end


userFeature = rand(numUser, numFeature) * 0.1;
movieFeature = rand(numMovie, numFeature) *  0.1;

training = 1;

loopCondition = true;


while loopCondition
    
    RMSE(training) = 0;
    predRatings = userFeature * movieFeature';
    
    
    for i=1:length(userID)
        
        tempMovieFeature = movieFeature(movieID(i),:);
        predictionForLoop = globalAverage + biasMovie(movieID(i)) + biasUser(userID(i)) + predRatings(userID(i), movieID(i));
        
        error = rating(i) - predictionForLoop;
        
        movieFeature(movieID(i),:) = movieFeature(movieID(i),:) + learningRate* ( error * userFeature(userID(i),:)-regulationRate*movieFeature(movieID(i),:));
        userFeature(userID(i),:) = userFeature(userID(i),:) + learningRate * (error * tempMovieFeature - regulationRate*userFeature(userID(i),:));
        
        RMSE(training) = RMSE(training) + error^2;
        
    end
    
    RMSE(training) = RMSE(training)/length(userID);
    
    disp(RMSE(training))
    
    plot(1:training,RMSE);
    drawnow;
    
    if training > 10 && RMSE(training)/RMSE(training-1) > 0.9999999
        loopCondition = false;
    end
    
    if training > 100
        loopCondition = false;
    end
    
    training = training + 1;
    
end

userTSNE = tsne(userFeature, [], 2);
figure, scatter(userTSNE(:,1),userTSNE(:,2))

movieTSNE = tsne(movieFeature, [], 2);
figure, scatter(movieTSNE(:,1),movieTSNE(:,2))

csvwrite('movieTSNE.csv',movieTSNE);
csvwrite('userTSNE.csv',userTSNE);

userTSNENor = (userTSNE + 60)/120.0
movieTSNENor = (movieTSNE+60) /120

csvwrite('movieTSNENor.csv',movieTSNENor);
csvwrite('userTSNENor.csv',userTSNENor);













