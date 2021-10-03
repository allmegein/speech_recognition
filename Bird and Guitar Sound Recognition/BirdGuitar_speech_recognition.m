%% AUTOMATIC SPEECH RECOGNITION USING MEL FREQUENCY CEPSTRAL COEFFICIENTS %%
% 22.06.2019 - allmegein
%% PRELIMINARY
clear;
close all;
clc;

folder = 'your path';
audiofile = dir(fullfile(folder , '*.wav'));
tic;
fprintf('The audio files are being read...');
traininput = [];
for f = 1 : numel(audiofile)
filename = audiofile(f).name;

% filename = 'your path';

% define parameters
frameduration = 0.010; %frame duration(ms)
overlapped = frameduration * 0.75; % starting point of next frame(ms). %25 overlapped.

% read to audio file.
[rawspeech , fs] = audioread(filename);
% sound(rawspeech,fs);

% enter new frequency rate that you want and resample the signal.
fns = 16000; % 16 KHz sample rate.
if fs > fns
fstep = round(fs / fns);
Nf = length(rawspeech);
new_signal = zeros(1 , floor(length(rawspeech)/round(fs/fns)));
ctr = 1;
for v = 1 : Nf
    if mod(v , fstep) == 0
        new_signal(ctr) = rawspeech(v);
        ctr = ctr + 1;
    end
end
    new_signal = new_signal';
else
    new_signal = rawspeech;
end

% reshape the original signal and keep as N x 1 column vector.
[srow , scol] = size(new_signal);
if scol > srow
    new_signal = new_signal'; % guarantee as a column vector.
end

% normalization of the original signal.
speech = new_signal(: , 1)/max(abs(new_signal(: , 1)));

% assign very small number instead of zeros to avoid error dividing by zero.
speech(speech == 0) = 1e-9;

% plotting the original and normalized signal.
figure(1);
subplot(211);
plot(rawspeech(:,1));
title('Original Speech');
xlabel('frequency(samples)');ylabel('amplitude');
subplot(212);
plot(speech);
title('Normalized Speech');
xlabel('frequency(samples)');ylabel('amplitude');

%% Pre-emphasising
% This step is used for boosting of high frequency component energy.
p_filter_values = [1 -0.97];
p_speech = filter(p_filter_values , 1 , speech);

% plotting the speech signal after applied preemphasis filter.
figure(2);
plot(p_speech);
title('Pre-emphased Signal');xlabel('Samples');ylabel('Amplitude');

%% FRAMING
% Framing is very important part of signal processing. It splits up the signal into the frames.
% This step must be applied because the original signal goes to infinity.So, the signal must be analysed frame by frame.

N = length(p_speech); % length of speech signal.
framelength = round(frameduration * fns); % frame length.
step = round(overlapped * fns); % starting point of step.
nStep = ceil(N / step); % this variable gives us how many frames we have.
[p_speech] = [p_speech ; zeros(framelength , 1)]; % zero padding.
framesMatrix = zeros(nStep , framelength);
for i = 1 : nStep
    startPoint = (i-1)*step + 1; % frame starting point
    endPoint = startPoint + framelength - 1; %frame end point
    framesMatrix(i , :) = p_speech(startPoint : endPoint); % creating a new matrix to store frames.
end

%% WINDOWING (HAMMING)

[sRow , sCol] = size(framesMatrix); % size of frameMatrix matrix.sRow = no. of frames.

window = hamming(sCol); % applying hamming window.

%plotting window function
% figure(3);
% plot(window); title('Hamming Window');
% xlabel('samples');ylabel('amplitude');

% applying hamming window for each frame.
windowMatrix = zeros(sRow , sCol);
for k = 1 : sRow
    windowMatrix(k , :) = framesMatrix(k , :) .* window'; % and store it into a new matrix which named windowMatrix.
end

%% DISCRETE TIME FOURIER TRANSFORM (DTFT / FFT)
% Take the DTFT to analyze the signal by transfering it from time domain to frequency domain.

nfft = 2^nextpow2(frameduration * fns); % length of dtft
PSDLength = nfft / 2 + 1; % half of PSD length cause of fft's symmetry.

% fast fourier transform
dtft = zeros(sRow , nfft);
for j = 1 : sRow
    dtft(j , :) = abs(fft(windowMatrix(j , :) , nfft)); % magnitude info.
end

%% POWER SPECTRUM (PSD)
PSD = dtft.^2;
PSD = PSD(: , 1 : PSDLength);

% plotting one frame fft example
plot(dtft(112 , :));

%% MEL FILTER BANK
% creating mel filters

% define parameters that used to create Mel filters.
lowfreq = 300; % low frequency of Mel filter bank(Hz).
highfreq = fns / 2; % high frequency of Mel filter bank(Hz). It has to be limited below samplerate/2;
nomelf = 26; % no. of Mel filters.
nofdctcoeff = 12; % no. of coefficients after applying dct.

% m = 1125xln(1+f/700); % f = 700xe^((m/1125)-1); % mel scale mathematical formula
lowMel = 1125 * log(1 + lowfreq /700); % converting Hertz value to Mel scale for lower frequency value.
highMel = 1125 * log(1 + highfreq /700); % converting Hertz value to Mel scale for higher frequency value.

filtersteps = (highMel - lowMel) / (nomelf + 1); % filter steps at Mel values.

melFreq = zeros(1 , nomelf + 1);
melFreq(1) = lowMel; % assign the low Mel frequency to first cell.
for m = 2 : nomelf + 1 % if we want to create N filters, we have to spesify N+2 point.
    melFreq(m) = lowMel + (m - 1) * filtersteps; % do it for each cell to create filters.
end

melFreq(end + 1) = highMel;

% convert the Mel frequencies back to Hertz.
% Thanks to this step , the frequency filter steps has been obtained.

BackToHertz = 700 * (exp(melFreq /1125) - 1); % back to Hertz
samplesHertz = floor(nfft * BackToHertz / fns); % each filter samples at Hertz.

% creating Mel filters.
% This part contains mathematical formula of the mel filters.
MelMatrix = zeros(nomelf , PSDLength);
for i = 2 : nomelf + 1
    for k = 1 : PSDLength
        if (k < samplesHertz(i - 1))
            MelMatrix(i-1 , k) = 0;
        else
            if (samplesHertz(i - 1) <= k && k <= samplesHertz(i))
                MelMatrix(i - 1 , k) = (k - samplesHertz(i - 1)) / (samplesHertz(i) - samplesHertz(i - 1));
            else
                if (samplesHertz(i) <= k  && k <= samplesHertz(i + 1))
                    MelMatrix(i - 1 , k) = (samplesHertz(i + 1) - k) / (samplesHertz(i + 1) - samplesHertz(i));
                else
                    MelMatrix(i - 1 , k) = 0;
                end
            end
        end
    end
end

MFilterB = MelMatrix;

%plotting Mel filter bank.
figure(4);
plot(1 : PSDLength , MFilterB);
title('Mel Filter Bank');
xlabel('Frequency(Hz)');ylabel('Amplitude');

%% ENERGY
% calculate the energy of each frames by passing into all of filters.

melEnergies = zeros(1 , PSDLength);
melScalarEnergies = zeros(nomelf , sRow);
for frc = 1 : sRow
    for fic = 1 : nomelf
         melEnergies(fic , :) = MFilterB(fic , :) .* PSD(frc, :); % energy calculation
         melScalarEnergies(fic , frc) = sum(melEnergies(fic , :)); % scalar energy of each frame at each filter. There are 26 filters and its keep each frame's total energy.
    end
end
melLogEnergies = log(melScalarEnergies); % logarithm of mel energies.

%% DISCRETE COSINE TRANSFORM (DCT)
% To keep important lower 12 coefficients to upgrade ASR performance.

melCoeffs = zeros(sRow , nofdctcoeff);
for h = 1 : sRow
    for j = 1 : nofdctcoeff
        cepstralSum = 0;
        for n = 1 : nomelf
            cepstralSum = cepstralSum + melLogEnergies(n , h) .* cos((pi * j / nomelf) * (n - 0.5)); % DCT /Inverse DTFT formula.
        end
        melCoeffs(h , j) = cepstralSum .* sqrt(2 / nomelf);
    end
end

%% add frame energies next to the coefficients matrix to improve the ASR performance.

frameEnergies = zeros(sRow , 1);
for w = 1 : sRow
    frameEnergies(w) = sum(windowMatrix(w , :).^2); % calculate each frame energy.
end

featureMatrix = [melCoeffs frameEnergies];

[traininput] = [traininput ; featureMatrix];

end

fprintf('\nCompleted\n');
% labeling as 1,0
[FR , CO] = size(traininput);
traininput(1 : FR/2 , end+1) = 1; % label as 1.
traininput(FR/2 + 1 , end) = 0; % label as 0.

xlswrite('Excel File of Bird and Guitar.xlsx' , traininput); % create an Excel file.
toc;

%% SINGLE LAYER PERCEPTRON %%

% define initial parameters and train.
ip = traininput; % input of data for training
[R , C] = size(ip);
numIn = C - 1; % number of input

desired_output = ip(: , end); % desired output
bias = 1; % bias
alpha = 0.5; % learning rate
weights = zeros(numIn + 1 , 1); % weights
iteration = 10;
out = zeros(R , 1);
for it = 1 : iteration

    for tr = 1 : R

        y = bias * weights(1,1) + ip(tr,1:end-1) * weights(2:end);

        out(tr) = 1 / (1 + exp(-y)); % sigmoid activation function.

        error = desired_output(tr) - out(tr); % error (delta rule)

        % update the weights...
        weights(1,1) = weights(1,1) + alpha * bias * error
        weights(2:end) = weights(2:end) + alpha * ip(tr,1:end-1)' * error

    end
end
fprintf('End of Training.\n');
xlswrite('Updated Weights of MFCC' , weights); % store the updated weights.

% Report will be written for describe the mentality of MFCC algorithm and single layer perceptron.
%% -------------------------------------------------------------------------------------------------
