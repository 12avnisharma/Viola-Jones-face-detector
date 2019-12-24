function [newWeights,alpha] = adaboost(classifier, images, imgWeights)
imgsSize = 507+1860; % total number of images in MIT database
faceSize = 507; % number of face images
captures = zeros(imgsSize,1);
error = 0;

for i = 1:imgsSize
    img = images{i};
    % obtains classifier metadata from fields in the row vector 
    haar = classifier(1);
    pixelX = classifier(2);
    pixelY = classifier(3);
    haarX = classifier(4);
    haarY = classifier(5);
    % calculates intensity difference between black-white region of the
    % Haar feature and checks against the precalculated range
    haarVal = calculateHaarVal(img,haar,pixelX,pixelY,haarX,haarY);
    if haarVal >= classifier(9) && haarVal <= classifier(10) % if falls between correct value
        if i <= faceSize % if its a face
            captures(i) = 1; % correct capture
        else
            captures(i) = 0; % error
            error = error + imgWeights(i); % increase weighted error count
        end
    else % if falls outside the expected range
        if i <= faceSize % if is a face
            captures(i) = 0;
            error = error + imgWeights(i); % error
        else 
            captures(i) = 1;
        end
    end
end

alpha = 0.5*log((1-error)/error); % updates classifier weight (alpha)


for i = 1:imgsSize
    if captures(i) == 0
        imgWeights(i) = imgWeights(i).*exp(alpha);
    else
        imgWeights(i) = imgWeights(i).*exp(-alpha);
    end
end
imgWeights = imgWeights./sum(imgWeights); % normalize image weights
newWeights = imgWeights; % pass as function output
end