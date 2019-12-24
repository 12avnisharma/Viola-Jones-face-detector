function intensity = getCorners(img,startX,startY,endX,endY)
    a = img(startY,startX);
    b = img(startY,endX);
    c = img(endY,startX);
    d = img(endY,endX);
    intensity = d-(b+c)+a; % by property of the integral image
end