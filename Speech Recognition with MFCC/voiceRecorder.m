%% audio recorder 44100 Hz,16 Bits,mono channel
% 1 second sound record and write it to replay outside of the Matlab.

recObj = audiorecorder(44100,16,1);

prompt='Press Enter to record a 1 second speech\n';
recorderbutton=input(prompt,'s');
disp('Start speaking...');
recordblocking(recObj,1);
disp('End of Recording.');

play(recObj);

y2 = getaudiodata(recObj);

filename='l1.wav';
audiowrite(filename,y2,44100);
