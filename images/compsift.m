%Computing Sift Descriptors of sets of images.


files = dir('PainterFemale\*.jpg');
no=0;
for file = files'
    I = imread(strcat('PainterFemale\',file.name));
    file.name
    I = single(rgb2gray(I)) ;
    [f,d] = vl_sift(I);
    dlmwrite(strcat('PainterFemale\',int2str(no),'.txt'),d');
    no=no+1;
end

files = dir('PainterMale\*.jpg');
no=0;
for file = files'
    I = imread(strcat('PainterMale\',file.name));
    file.name
    I = single(rgb2gray(I)) ;
    [f,d] = vl_sift(I);
    dlmwrite(strcat('PainterMale\',int2str(no),'.txt'),d');
    no=no+1;
end



files = dir('ScientistFemale\*.jpg');
no=0;
for file = files'
    I = imread(strcat('ScientistFemale\',file.name));
    file.name
    I = single(rgb2gray(I)) ;
    [f,d] = vl_sift(I);
    dlmwrite(strcat('ScientistFemale\',int2str(no),'.txt'),d');
    no=no+1;
end

files = dir('ScientistMale\*.jpg');
no=0;
for file = files'
    I = imread(strcat('ScientistMale\',file.name));
    file.name
    I = single(rgb2gray(I)) ;
    [f,d] = vl_sift(I);
    dlmwrite(strcat('ScientistMale\',int2str(no),'.txt'),d');
    no=no+1;
end
