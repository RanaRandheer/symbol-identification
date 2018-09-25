import numpy as np

a = np.genfromtxt('P1_data_train.csv',delimiter=',');
b = np.genfromtxt('P1_labels_train.csv',delimiter=',');

mean5=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
       0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
       0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];
mean6=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
       0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
       0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];

m5=np.array([mean5]);
m6=np.array([mean6]);
var5=np.zeros((64,64));
var6=np.zeros((64,64));

n5,n6=0,0;

for i in range(777):
    if b[i]==5:
        n5=n5+1
        mean5=mean5+a[i];
    if b[i]==6:
        mean6=mean6+a[i];

n6=777-n5;
mean5,mean6=mean5/n5,mean6/n6;

for i in range(777):
    x=np.array([a[i]])
    if b[i]==5:
        var5=var5+np.dot((x-m5).T,(x-m5));
    if b[i]==6:
        var6=var5+np.dot((x-m6).T,(x-m6));

var5,var6=var5/n5,var6/n6;

det5=np.linalg.det(var5);
det6=np.linalg.det(var6);

inv5=np.linalg.inv(var5);
inv6=np.linalg.inv(var6);

g1=np.log(n5/n6)+0.5*(np.log(det6/det5));

a1 = np.genfromtxt('P1_data_test.csv',delimiter=',');
b1 = np.genfromtxt('P1_labels_test.csv',delimiter=',');

no5=0;
for i in range(333):
    if b1[i]==5:
        no5=no5+1;

acc1=0;acc2=0; 

for i in range(333):
    t5=np.array([a1[i]])
    t5=t5+m5
    t6=t5+m6
    g2=0.5*( np.dot(np.dot(t6,inv6),t6.T)-np.dot(np.dot(t5,inv5),t5.T) )
    if (g1+g2)>0 and b1[i]==5:
        acc1=acc1+1
    if (g1+g2)<0 and b1[i]==6:
        acc2=acc2+1
    
print (np.array([[acc1,no5-acc1],[333-no5-acc2,acc2]]));
