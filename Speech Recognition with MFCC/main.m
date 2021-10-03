%% AUTOMATIC SPEECH RECOGNITION USING MEL FREQUENCY CEPSTRAL COEFFICIENTS
% 29.02.2020 - allmegein
%% PRELIMINARIES
clear,close all;
clc;

% read to audio file.
folder = 'your path';
filename = dir(fullfile(folder,'*.wav'));
featureMatrix = [];
disp('All sounds are being loaded...');
for it = 1:numel(filename)
[rawspeech , fs] = audioread(filename(it).name);
figure(1),plot(rawspeech),xlabel('time'),ylabel('magnitude'),title('Speech Signal');

% define parameters
fd = 0.01; %frame duration(ms)
fshift = fd * 0.20; % frame overlap

% normalization of the original signal.
speech = rawspeech(: , 1)/max(abs(rawspeech(: , 1)));
figure(2),plot(speech),xlabel('time'),ylabel('magnitude'),title('Normalized Speech Signal');

%% Pre-emphasising
% This step is used for boosting of high frequency component energy.
pfValue = [1 -0.97];
pSpeech = filter(pfValue , 1 , speech);

% Compare pre-emphasized and original speech out loud.
% sound(speech,fs);
% sound(pSpeech,fs);

%% FRAMING
% Framing is very important part of signal processing. It splits up the signal into the frames.
% This step must be applied because a voice signal goes to infinity.So, the signal must be analysed frame by frame.
% Spectral Analysis -

N = length(pSpeech); % length of speech signal.
framelength = round(fd * fs); % frame length.
step = round(fshift * fs); % starting point of step.
nStep = ceil(N / step); % no. of frames.
[pSpeech] = [pSpeech ; zeros(framelength , 1)]; % zero padding.
fMatrix = zeros(nStep , framelength);
for i = 1 : nStep
    startPoint = (i-1)*step + 1; % starting point
    endPoint = startPoint + framelength - 1; % ending point
    fMatrix(i , :) = pSpeech(startPoint : endPoint); % creating a matrix to store frames.
end

%% WINDOWING

[sRow , sCol] = size(fMatrix); % size of frameMatrix matrix.sRow = no. of frames.

window = hamming(sCol); % applying hamming window.
figure(3),subplot(211),plot(window),xlabel('time'),ylabel('magnitude'),title('Hamming Window');

% applying hamming window for each frame.
windowMatrix = zeros(sRow , sCol);
for k = 1 : sRow
    windowMatrix(k , :) = fMatrix(k , :) .* window'; % and store it into a new matrix which named windowMatrix.
end
subplot(212),plot(fMatrix(20,:)),hold on,plot(windowMatrix(20,:)),hold off,xlabel('time'),ylabel('magnitude'),title('A part of Signal Example after Hamming window');
%% DISCRETE TIME FOURIER TRANSFORM (DTFT / FFT)
% Take the DTFT to analyze the signal by transfering it from time domain to frequency domain.

nfft = 2^nextpow2(fd * fs); % length of dtft
PSDLength = nfft / 2 + 1; % half of PSD length because fft is double-sided.(single side of fft).

% fast fourier transform
dtft = zeros(sRow , nfft);
for j = 1 : sRow
    dtft(j , :) = abs(fft(windowMatrix(j , :) , nfft)); % magnitude info.
end

%% POWER SPECTRUM (PSD)
PSD = dtft.^2;
PSD = PSD(: , 1 : PSDLength);
% plotting one frame example of fft and psd.
% figure,plot(dtft(150,:)),hold on,plot(PSD(150,:));

%% MEL FILTER BANK

% define parameters
lowfreq = 300; % low frequency of Mel filter bank(Hz).
highfreq = fs / 2; % high frequency of Mel filter bank(Hz). It has to be limited below samplerate/2;
nof = 26; % no. of Mel filters.
noc = 12; % most important no. of cepstral coefficients.

lowMel = 1125 * log(1 + lowfreq /700); % converting Hertz value to Mel scale for lower frequency value.
highMel = 1125 * log(1 + highfreq /700); % converting Hertz value to Mel scale for higher frequency value.

fSteps = (highMel - lowMel) / (nof + 1); % filter steps at Mel values.

melFreq = zeros(1 , nof + 1);
melFreq(1) = lowMel; % assign the low Mel frequency into first cell.
for m = 2 : nof + 1 % if we want to create N filters, we have to spesify N+2 point.
    melFreq(m) = lowMel + (m - 1) * fSteps; % do it for each cell to create filters.
end

melFreq(end + 1) = highMel;

% convert the Mel frequencies back to Hertz
BackToHertz = 700 * (exp(melFreq /1125) - 1); % back to Hertz.
samplesHertz = floor(nfft * BackToHertz / fs); % each filter samples at Hertz.

% creating Mel filters.
MelMatrix = zeros(nof , PSDLength);
for i = 2 : nof + 1
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

% plotting Mel filter bank
figure(4),plot(1:PSDLength,MFilterB),xlabel('frequency'),ylabel('amplitude'),title([num2str(nof),' Mel-Filterbank']);

%% ENERGY
% calculate the energy of each frames by passing into all of filters.

Energies = zeros(1 , PSDLength);
melEnergies = zeros(nof , sRow);
for frc = 1 : sRow
    for fic = 1 : nof
         Energies(fic , :) = MFilterB(fic , :) .* PSD(frc, :); % energy calculation
         melEnergies(fic , frc) = sum(Energies(fic , :)); % scalar energy of each frame at each filter. There are 26 filters and its keep each frame's total energy.
    end
end

% take logarithm of mel energies.
logE = log(melEnergies);

%% DISCRETE COSINE TRANSFORM (DCT)
% To keep important 12 coefficients to upgrade ASR performance.

melCoeffs = zeros(sRow , noc);
for h = 1 : sRow
    for j = 1 : noc
        cepSum = 0;
        for n = 1 : nof
            cepSum = cepSum + logE(n , h) .* cos((pi * j / nof) * (n - 0.5)); % DCT /Inverse DTFT formula.
        end
        melCoeffs(h , j) = cepSum .* sqrt(2 / nof);
    end
end

%% add frame energies next to the cepstral coefficient matrix to improve the ASR performance.

frameEnergies = zeros(sRow , 1);
for w = 1 : sRow
    frameEnergies(w) = sum(windowMatrix(w , :).^2); % calculate each frame energy.
end

features = [melCoeffs frameEnergies];
[featureMatrix] = [featureMatrix ; features];

end


% dataset labelling to create class for letters.(0-1-2).
featureMatrix(1:length(featureMatrix)/3,end+1)=0;                               % A letter label
featureMatrix(length(featureMatrix)/3+1:2*length(featureMatrix)/3,end)=1;       % I letter label
featureMatrix(2*length(featureMatrix)/3+1:3*length(featureMatrix)/3,end)=2;     % L letter label

figure(5),plot(featureMatrix(1:length(featureMatrix)/3,1:end-1),'r.');
hold on,plot(featureMatrix(length(featureMatrix)/3+1:2*length(featureMatrix)/3,1:end-1),'b.');
hold on,plot(featureMatrix(2*length(featureMatrix)/3+1:3*length(featureMatrix)/3,1:end-1),'g.');
xlabel('samples'),ylabel('magnitude'),title("Distribution of A-I-L Letters");

disp('Completed.');

% transfer to Excel
xlswrite('MFCC Voice Data.xls' , featureMatrix);
disp('Excel file was created.');
