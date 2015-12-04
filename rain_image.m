load raindata
out_dir='/Users/joshuamannheimer/documents/MATLAB/rain_images';
data=raindata;
sz=size(data);
a=[];
b=[];
a=[a,1];
for i=2:sz(1)
    if data(i,2)~=data(i-1,2)
        a=[a,i];
        b=[b,i-1];
    end
end
b=[b,i];
fname=sprintf('Rain_Images_reflectivity_%d.jpg',i');
for i=1:numel(a)
    im_data=data(a(i):b(i),4:8);
    t=data(a(i):b(i),3);
    x1=im_data(:,1);
    X1=((x1-max(x1))+(x1-min(x1)))./(max(x1)-min(x1));
    t=t./60.0;
    theta1=t.*acos(X1);
    theta2=theta1*theta1';
    M=real((eye(numel(x1))-theta2)^(1/2));
    G=theta2-M'*M;
    Z=zeros(23);
    L=numel(X1);
    Z(3:(L+2),3:(L+2))=G;
    im=mat2gray(G);
    Z(3:(L+2),3:(L+2))=im;
    
    fname=sprintf('Rain_Images_reflectivity_%d.png',i');
    imwrite(Z,fullfile(out_dir,fname))
    if size(Z)~=[23,23]
        print('error')
    end
    
end
rain_vals=data(a,9);
csvwrite('rain_measures.csv',rain_vals)

