
%%
clc;clear;
cmap=load("cmap.mat").colormap;
SamplePath1 = 'SegmentationClassPNG';  %存储图像的路径
SamplePath2 = 'JPEGImages';
SamplePath3 = 'JPEGImagesEdge';
SamplePath4 = 'SegmentationClassPNGEdgeMedian';
SamplePath5= '最后七张原图边缘检测';
fileExt = '*.png';  %待读取图像的后缀名
%获取所有路径
files = dir(fullfile(SamplePath1,fileExt)); 

pixfiles = dir(fullfile(SamplePath2,fileExt)); 
len1 = size(files,1);
%遍历路径下每一幅图像
for i=1:len1;
   fileName = strcat(SamplePath1,'/',files(i).name); 

   pixfilename=strcat(SamplePath2,'/',pixfiles(i).name); 

   writefilename=strcat(SamplePath5,'/',pixfiles(i).name); 

   % %原图边缘检测
   pix=imread(pixfilename);
   img=rgb2gray(pix);

   Log=edge(img, 'Sobel');
   kernel = 2;
   Log= medfilt2(Log,[kernel ,kernel ])*255;

   f = im2double(pix); 
   pp(:,:,1)=f(:,:,1)+Log;
   pp(:,:,2)=f(:,:,2)+Log;
   pp(:,:,3)=f(:,:,3)+Log;

   imwrite(pp,writefilename);


%    img = imread(fileName);
%    Log=edge(img, 'canny');
%    kernel = 2;
%    Log= medfilt2(Log,[kernel ,kernel ])*255;
% % % label加边缘
%     f=im2double(img);
%     newpix =im2uint8(f+Log);
%     imwrite(newpix,writefilename);
   
   %canny算子
% %  原图加边缘
%    pix=imread(pixfilename);
%    f = im2double(pix); 
%    pp(:,:,1)=f(:,:,1)+Log;
%    pp(:,:,2)=f(:,:,2)+Log;
%    pp(:,:,3)=f(:,:,3)+Log;
% 
%    newpix = im2uint8(pp);

   %I_gray=rgb2gray(img);
   %imwrite(newpix,writefilename);
   
   
end


%%
clc;clear
pix=imread('./JPEGImages\img00000034.png');
img=rgb2gray(pix);

Log=edge(img, 'Sobel');

kernel = 2;
Log= medfilt2(Log,[kernel ,kernel ]);
%imwrite(Log,'./Edge00.png');
imshow(Log)

