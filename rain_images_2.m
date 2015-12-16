load NEW_rdata
data=r_data;
load A_B_VALS
a=a(:,2);
b=b(:,3);
as=[];
bs=[];
cs=b-a;
for i=1:23000
    if cs(i)>5
        as=[as a(i)];
        bs=[bs b(i)];
    end
end
a=as;
b=bs;
out_dir='/Users/joshuamannheimer/documents/MATLAB/rain_RBG';
c=b-a;
min_val=min(c)+1;
for i=1:1
    fname=sprintf('Rain_Images_%d.png',i');
    im_data=data(a(i):b(i),5:9);
    t=[data(a(i):b(i),4);data(a(i):b(i),4);data(a(i):b(i),4);data(a(i):b(i),4);data(a(i):b(i),4)]./60;
    x1=im_data(:,1);
    x2=im_data(:,2);
    x3=im_data(:,3);
    x4=im_data(:,4);
    x5=im_data(:,5);
    
    mx1=max(x1);
    mn1=min(x1);
    mx2=max(x2);
    mn2=min(x2);
    mx3=max(x3);
    mn3=min(x3);
    mx4=max(x4);
    mn4=min(x4);
    mx5=max(x5);
    mn5=min(x5);
    
    
    x1_0=((x1-mx1)+(x1-mn1))./(mx1-mn1);
    x2_0=((x2-mx2)+(x2-mn2))./(mx2-mn2);
    x3_0=((x3-mx3)+(x3-mn3))./(mx3-mn3);
    x4_0=((x4-mx4)+(x4-mn4))./(mx4-mn4);
    x5_0=((x5-mx5)+(x5-mn5))./(mx5-mn5);
    X=[x1_0;x2_0;x3_0;x4_0;x5_0];
    
    vec=linspace(1,numel(X),numel(X));
    ran_choice=randsample(numel(X),(min_val*5));
    idx=sort(ran_choice);
    Z=X(idx);
    theta1=t(idx).*acos(Z);
    theta2=theta1*theta1';
    M=real((eye(numel(Z))-theta2)^(1/2));
    G=theta2-M'*M;
    Heat(G)
    %im=mat2gray(G);
    if size(im)~=[35,35]
        
    end
    imwrite(im,fullfile(out_dir,fname))
    
end
