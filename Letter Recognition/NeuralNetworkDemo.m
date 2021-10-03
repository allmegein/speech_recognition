%% SPEECH RECOGNITION WITH ANN %%
% allmegein - 14.03.2020 %
% ----------------------------------------------------------------------- %
% This is the speech recognition script which is classified the data by
% using Artificial Neural Network. Only the magnnitude information of audio
% is used. Labels are as shown follows :
%                       [0 0 1] : A
%                       [0 1 0] : �
%                       [1 0 0] : L

% This file is linked to the file named ( NeuralNetworkTest.m ).
% ----------------------------------------------------------------------- %
clear, close all;
clc;

% read all audio files
folder = 'your path';
filename = dir(fullfile(folder,'*.wav'));
A_mag_info = [];
for f = 1:numel(filename)

[y,fs] = audioread(filename(f).name);

y_normalized = y / (max(abs(y)));        % normalization

mag = abs(fft(y_normalized));            % magnitude info of audio
% dtft(k,:) = mag(1:length(mag)/2+1);
dtft = mag(1:length(mag)/2+1);

[A_mag_info] = [A_mag_info , dtft];      % write all audios below

end

A_mag_info = A_mag_info';

%% Neural Network Classifier
load('A_mag_info.mat');                   % load the file
data = A_mag_info(:,1:end-3);
data = [ones(size(data,1),1) data];       % add bias term for all audios
d_out = A_mag_info(:,end-2:end);          % desired output (labels)

W=zeros(length(data) , 3);        % weights
alp=0.02;                         % learning rate
iteration = 100;                  % iteration number


for i=1:iteration
    for k=1:size(data , 1)

        y = data(k,:) * W;                      % input of a neuron
        out = 1 ./ (1 + exp(-y));               % output of a neuron after sigmoid activation function

        error = d_out(k,:) - out;               % error term
%         e(k,:) = error;
        W = W + alp * data(k,:)' .* error;      % update the weigths


    end


end

save wei.mat W;                 % store weights as dataset

for j=1:18
    plot(A_mag_info(j,:)),hold on,
end

plot(A_mag_info(1,:)),hold on;
plot(A_mag_info(6,:)),hold on;
plot(A_mag_info(12,:)),hold on;
