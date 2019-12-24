
clc;
clear all;
faceSize = 507;
nonFaceSize = 1860;
faces = cell(1,faceSize);
nonFaces = cell(1,nonFaceSize);
temp1 = []; temp2=[]; temp3=[]; temp4=[]; temp5=[];
fprintf('Reading Face Images\n');
for faceNum = 1:faceSize
    str = 'C:\Users\avnis\OneDrive - Michigan State University\Desktop\CVHW3\New folder\TrainingFaces\';
    img = int2str(faceNum);
    fullPath = strcat(str,img,'.png');
    img = imread(fullPath);
    integral = integralImg(img);
    faces{faceNum} = integral;
end
 allImages = faces;
fprintf('Reading Non-Face Images\n');
for nonFaceNum = 1:nonFaceSize
    str = 'C:\Users\avnis\OneDrive - Michigan State University\Desktop\CVHW3\New folder\TrainingNonFaces\';
    img = int2str(nonFaceNum);
    fullPath = strcat(str,img,'.png');
    img = imread(fullPath);
    integral = integralImg(img);
    nonFaces{nonFaceNum} = integral;
    allImages{nonFaceNum+faceSize} = integral;
end   
fprintf('Constructing Haar Features\n');

imgWeights = ones(faceSize+nonFaceSize,1)./(faceSize+nonFaceSize);
haars = [1,2;2,1;1,3;3,1;2,2];
window = 24;
    weakClassifiers = {};
    for haar = 1:5
        printout = strcat('Working on Haar #',int2str(haar),'\n');
        fprintf(printout);
      
        dimX = haars(haar,1);
    
        dimY = haars(haar,2);
   
        for pixelX = 2:window-dimX
            for pixelY = 2:window-dimY
                for haarX = dimX:dimX:window-pixelX
                    for haarY = dimY:dimY:window-pixelY
                        haarVector1 = zeros(1,faceSize);
                        for img = 1:faceSize
                            [val]= calculateHaarVal(faces{img},haar,pixelX,pixelY,haarX,haarY);
                            haarVector1(img) = val;
                        end
                        faceMean = mean(haarVector1);
                        faceStd = std(haarVector1);
                        faceMax = max(haarVector1);
                        faceMin = min(haarVector1);
                        haarVector2 = zeros(1,nonFaceSize);
                        
                        for img = 1:nonFaceSize
                           
                            % image
                            val = calculateHaarVal(nonFaces{img},haar,pixelX,pixelY,haarX,haarY);
                   
                            haarVector2(img) = val;
                        end
                   
  
                        storeRatingDiff = [];
                        storeFaceRating = [];
                        storeNonFaceRating = [];
                        storeTotalError = [];
                        storeLowerBound = [];
                        storeUpperBound = [];
                        strongCounter = 0;
                     
                        for iter = 1:50
                            C = ones(size(imgWeights,1),1);
                            minRating = faceMean-abs((iter/50)*(faceMean-faceMin));
                            maxRating = faceMean+abs((iter/50)*(faceMax-faceMean));
                            
                            for val = 1:faceSize
                                if haarVector1(val) >= minRating && haarVector1(val) <= maxRating
                                    C(val) = 0;
                                end
                            end
                            faceRating = sum(imgWeights(1:faceSize).*C(1:faceSize));
                            if faceRating < 0.05 % if less than 5% faces misclassified
                                % capture all false positive values
                                for val = 1:nonFaceSize
                                    if haarVector2(val) >= minRating && haarVector2(val) <= maxRating
                                    else
                                        C(val+faceSize) = 0;
                                    end
                                end
                                % weighted false positive capture rate
                                nonFaceRating = sum(imgWeights(faceSize+1:nonFaceSize+faceSize).*C(faceSize+1:nonFaceSize+faceSize));
                                % total error
                                totalError = sum(imgWeights.*C);
                                if totalError < .5 % if less than 5% total error
                                    % store this as a weak classifier
                                    strongCounter = strongCounter+1;
                                    storeRatingDiff = [storeRatingDiff,(1-faceRating)-nonFaceRating];
                                    storeFaceRating = [storeFaceRating,1-faceRating];
                                    storeNonFaceRating = [storeNonFaceRating,nonFaceRating];
                                    storeTotalError = [storeTotalError,totalError];
                                    storeLowerBound = [storeLowerBound,minRating];
                                    storeUpperBound = [storeUpperBound,maxRating];
                                end
                            end
                        end

                       
                        if size(storeRatingDiff) > 0
                            maxRatingIndex = -inf; 
                            maxRatingDiff = max(storeRatingDiff);
                            for index = 1:size(storeRatingDiff,2)
                                if storeRatingDiff(index) == maxRatingDiff
                                    maxRatingIndex = index; 
                                    break;
                                end
                            end
                        end

                      
                        if size(storeRatingDiff) > 0
                            thisClassifier = [haar,pixelX,pixelY,haarX,haarY,maxRatingDiff,storeFaceRating(maxRatingIndex),storeNonFaceRating(maxRatingIndex),storeLowerBound(maxRatingIndex),storeUpperBound(maxRatingIndex),storeTotalError(maxRatingIndex)];

                       
                            [imgWeights,alpha] = adaboost(thisClassifier,allImages,imgWeights);
                            
                            thisClassifier = [thisClassifier,alpha];
                            
                            weakClassifiers{size(weakClassifiers,2)+1} = thisClassifier;
                           
                            if haar == 1
                                temp1 = [temp1; thisClassifier];
                            elseif haar == 2
                                temp2 = [temp2; thisClassifier];
                            elseif haar == 3
                                temp3 = [temp3; thisClassifier];
                            elseif haar == 4
                                temp4 = [temp4; thisClassifier];
                            elseif haar == 5
                                temp5 = [temp5; thisClassifier];
                            end
                        end
                    end
                end
            end
        end
        printout = strcat('Finished Haar #',int2str(haar),'\n');
        fprintf(printout);
    end 
%end


fprintf('Make strong classifiers from sorting according to alpha values\n');
alphas = zeros(size(weakClassifiers,2),1);
for i = 1:size(alphas,1)
   
    alphas(i) = weakClassifiers{i}(12);
end


tempClassifiers = zeros(size(alphas,1),2); 

tempClassifiers(:,1) = alphas;
for i = 1:size(alphas,1)
   tempClassifiers(i,2) = i; 
end

tempClassifiers = sortrows(tempClassifiers,-1); % sort descending order


selectedClassifiers = zeros(1,12);
for i = 1:1
    selectedClassifiers(i,:) = weakClassifiers{tempClassifiers(i,2)};
end


save('finalClassifiers.mat','selectedClassifiers');
