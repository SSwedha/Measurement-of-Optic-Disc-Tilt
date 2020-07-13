function [h, s, l] = rgb2hslown(image)
if(class(image) == 'double')
    ;
else
    image = double(image)/255;
end
[a,b,c] = size(image);

for i=1:a
    for j=1:b
        [maximum, whichcolor] = max(image(i,j,:));
        minimum = min(image(i,j,:));
        l(i,j) = (maximum + minimum) / 2;
        if(maximum == minimum)
            h(i,j) = 0;
            s(i,j) = 0;
        else
            d = maximum - minimum;
            if (l > 0.5)
                s(i,j) = d/(2-maximum-minimum);
            else
                s(i,j) = d/(maximum-minimum);
            end
            switch(whichcolor)
                case 1
                    if (image(i,j,2) < image(i,j,3))
                        something = 6;
                    else
                        something = 0;
                    end
                    h(i,j)=(image(i,j,2) - image(i,j,3)) / d + something;
                case 2
                    h(i,j) = (image(i,j,3) - image(i,j,1)) / d + 2;
                case 3
                    h(i,j) = (image(i,j,1) - image(i,j,2)) / d + 4;
            end
        end
        h(i,j) = h(i,j) / 6;
    end
end
end