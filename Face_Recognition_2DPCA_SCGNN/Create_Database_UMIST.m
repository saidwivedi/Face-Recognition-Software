function T = Create_Database_UMIST(ns,DatabasePath,ff,escImages)

count = 0;
T=[];
for td = 1:ns
       Files = dir(strcat(DatabasePath,int2str(td)));
for i = 1:size(Files,1)
  x=0;
    for j=1:length(escImages)
        if  (strcmp(Files(i).name,strcat(int2str(escImages(j)),ff))||strcmp(Files(i).name,strcat(int2str(escImages(j)),upper(ff))))
            x=1;
        else
            x=x;
        end
    end
    if not(strcmp(Files(i).name,'.')|strcmp(Files(i).name,'..')|strcmp(Files(i).name,'Thumbs.db')|x)
        img = imread(strcat(DatabasePath,int2str(td),'/',Files(i).name));
        if dim>1
            img=rgb2gray(img);
        end
        img = double(img)/255;
        count = count + 1; 
        T(count,:,:) = img;

    
    end
end

end
