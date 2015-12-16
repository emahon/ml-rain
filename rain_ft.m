oad NEW_rdata
data=r_data;
% Fromat Data to put into one D convolutional NN
times=data(:,4);
u_times=unique(times);
load A_B_VALS
a=a(:,2);
b=b(:,3);
as=[];
bs=[];
c=b-a+1;
for i=1:numel(a)
    if c(i)>=12
        as=[as,a(i)];
        bs=[bs,b(i)];
    end
end

a=as;
b=bs;
NEW_DATA=[];
for i=1:numel(a)
    n_data=data(a(i):b(i),5:9);
    t=data(a(i):b(i),4)+1;
    Z=zeros(60,5);
    Z(t,:)=n_data;
    count=1;
    L=numel(t);
    for j=1:60
        zs=sum(Z(j,:));
        if zs==0
            if count<=L
                Z(j,:)=n_data(count,:);
            else
                Z(j,:)=n_data((count-1),:);
            end
        else
            count=count+1;
        end
    end
    NEW_DATA=[NEW_DATA;Z];
end

guage_vals=data(a,10);
csvwrite('1D_Convolutional_Data.csv',NEW_DATA)
csvwrite('1D_guage_values.csv',guage_vals)
            
