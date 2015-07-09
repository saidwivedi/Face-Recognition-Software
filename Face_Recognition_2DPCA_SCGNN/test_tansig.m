clc;
clear all;
close all;
figure;

for x = -2:0.001:2;
    plot(x,tansig_approx(x),'r');
    plot(x,2/(1+exp(-2*x))-1,'b');
    %plot(x,((exp(x)-exp(-x))/(exp(x)+exp(-x))),'m');
    hold on;
end

