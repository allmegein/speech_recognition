%% TEST %%
% allmegein - 14.03.2020 %
% This file is linked to the file named ( NeuralNetworkDemo.m ).

clear, close all;
clc;

[y , fs] = voiceRecorder();

y_normalized = y / (max(abs(y)));

mag = abs(fft(y_normalized));
dtft = mag(1:length(mag)/2+1);
A_mag_info = dtft;
A_mag_info = A_mag_info';

data = A_mag_info;
data = [ones(size(data,1),1) data];

load('wei.mat');

y = data * W;
out = 1 ./ (1 + (exp(-y)));

% What did you say?
if out == [0 0 1]
    disp('You said (A).');
elseif out == [0 1 0]
    disp('You said (�).');
elseif out == [1 0 0]
    disp('You said (L)');
else
    disp('Your voice was not identified. Please try again.');
end
