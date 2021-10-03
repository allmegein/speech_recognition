clear all;
close all;
clc;

[m1,f1]=audioread('your path');
[m2,f2]=audioread('your path');
[m3,f3]=audioread('your path');
[m4,f4]=audioread('your path');
[m5,f5]=audioread('your path');
[m6,f6]=audioread('your path');
figure(1),plot(m1),figure(2),plot(m2);
figure(3),plot(m3),figure(4),plot(m4);
figure(5),plot(m5),figure(6),plot(m6);
