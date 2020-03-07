%{
    Jacob Ayers
    Program to convert .wav files into spectrograms.
    Also used to test what settings we want to tweak the spectrograms with.
    
    Note for the future: We may want to break down the audio files into
    smaller chunks as it may be difficult to grab anything meaningful from
    a time-axis that is an hour long. 
    
    In this snippet I pulled a 37s long clip from one of the wav files that
    had cricket chirps, branches falling/twigs being stepped on, and bugs
    buzzing close to the microphone
    
    Note I found from StackOverflow on altering the Window Variable

    The optimum window length will depend on your application.
    If your application is such that you need time domain information 
    to be more accurate, reduce the size of your windows.
    If the application demands frequency domain information to be more specific,
    then increase the size of the windows. As Hilmar mentioned, the Uncertainty
    Principle really leaves you with no other choice. You cannot get perfect resolution
    in both domains at once. You can get perfect resolution in only one domain 
    at the cost of zero resolution in the other (time and frequency domains) or 
    in-between resolution, but in both domains. 
    
    mathworks spectrogram information :
    https://www.mathworks.com/help/signal/ref/spectrogram.html#d118e187962

    %It was suggested by my grad student mentor Tim Woodford to set the
    number of fast fourier transforms (nfft) equal to the window size.
    %Try to figure out the differences between window types, I.E. Hanning
    vs. Hamming vs. inputting straight integer values.
    https://en.wikipedia.org/wiki/Window_function

    Useful tool to compare the different spectrograms that we cannot tell
    the difference of too keenly with our eyes.
    https://online-image-comparison.com/
%}
%Converting the wave file into a vector x, and frequency sampling rate fs
[x,fs]=audioread('TestRainforestSoundBite.wav');


figure('Name','Varying overlap sizes');
colormap(jet);
window2 = hamming(512);
fullTime = x(2:fs*37);
y = x(1:fs*5);
z = x(1:fs*20);
x = x(1:fs*10);







subplot(3,3,1);
spectrogram(x,window2,[],[],fs,'yaxis');
title('Default Overlap = 50%');

subplot(3,3,2);
spectrogram(x,window2,50,[],fs,'yaxis');
title('Overlap = 50');

subplot(3,3,3);
spectrogram(x,window2,100,[],fs,'yaxis');
title('Overlap = 100');

subplot(3,3,4);
spectrogram(x,window2,150,[],fs,'yaxis');
title('Overlap = 150');

subplot(3,3,5);
spectrogram(x,window2,[],[],fs,'yaxis');
title('Overlap = 200');

subplot(3,3,6);
spectrogram(x,window2,50,[],fs,'yaxis');
title('Overlap = 250');

subplot(3,3,7);
spectrogram(x,window2,[],[],fs,'yaxis');
title('Overlap = 300');

subplot(3,3,8);
spectrogram(x,window2,[],[],fs,'yaxis');
title('Overlap = 350');

subplot(3,3,9);
spectrogram(x,window2,[],[],fs,'yaxis');
title('Overlap = 400');
















figure('Name','Varying Window Sizes');

colormap(hot);

subplot(3,3,1);
spectrogram(x,100,[],[],fs,'yaxis');
title('Window = 100');

subplot(3,3,2);
spectrogram(x,600,[],[],fs,'yaxis');
title('Window = 600');

subplot(3,3,3);
spectrogram(x,1100,[],[],fs,'yaxis');
title('Window = 1100');

subplot(3,3,4);
spectrogram(x,1600,[],[],fs,'yaxis');
title('Window = 1600');

subplot(3,3,5);
spectrogram(x,2100,[],[],fs,'yaxis');
title('Window = 2100');

subplot(3,3,6);
spectrogram(x,2600,[],[],fs,'yaxis');
title('Window = 2600');

subplot(3,3,7);
spectrogram(x,3100,[],[],fs,'yaxis');
title('Window = 3100');

subplot(3,3,8);
spectrogram(x,3600,[],[],fs,'yaxis');
title('Window = 3600');

subplot(3,3,9);
spectrogram(x,4100,[],[],fs,'yaxis');
title('Window = 4100');














figure('Name','Varying # DFT Points');

colormap(bone);

subplot(3,3,1);
spectrogram(x,512,[],[],fs,'yaxis');
title('DFT = default');

subplot(3,3,2);
spectrogram(x,512,[],50,fs,'yaxis');
title('DFT = 50');

subplot(3,3,3);
spectrogram(x,512,[],100,fs,'yaxis');
title('DFT = 100');

subplot(3,3,4);
spectrogram(x,512,[],150,fs,'yaxis');
title('DFT = 150');

subplot(3,3,5);
spectrogram(x,512,[],200,fs,'yaxis');
title('DFT = 200');

subplot(3,3,6);
spectrogram(x,512,[],250,fs,'yaxis');
title('DFT = 250');

subplot(3,3,7);
spectrogram(x,512,[],300,fs,'yaxis');
title('DFT = 300');

subplot(3,3,8);
spectrogram(x,512,[],350,fs,'yaxis');
title('DFT = 350');

subplot(3,3,9);
spectrogram(x,512,[],400,fs,'yaxis');
title('DFT = 400');









figure('Name','Various approaches');
colormap(winter);

subplot(3,3,1);
spectrogram(x,512,[],512,fs,'yaxis');
title('Window = 512 = nfft, 50% overlap ');

subplot(3,3,2);
spectrogram(x,hamming(512),[],512,fs,'yaxis');
title('window = hamming(512)');

subplot(3,3,3);
spectrogram(x,hanning(512),[],512,fs,'yaxis');
title('window = hanning(512)');

subplot(3,3,4);
spectrogram(x,blackman(512),[],512,fs,'yaxis');
title('window = blackman(512)');

subplot(3,3,5);
spectrogram(x,flattopwin(512),[],512,fs,'yaxis');
title('window = flattopwin(512)');

subplot(3,3,6);
spectrogram(y,512,[],512,fs,'yaxis');
title('Decreasing Time Axis ');

subplot(3,3,7);
spectrogram(z,512,[],512,fs,'yaxis');
title('Increasing Time Axis');

subplot(3,3,8);
spectrogram(x,512,[],512,fs*2,'yaxis');
title('Doubling audioread frequency sampling');

subplot(3,3,9);
spectrogram(x,512,[],512,'yaxis');
title('Across full 37 seconds');


figure();
spectrogram(x,512,[],512,fs,'yaxis');
title('Window = 512 = nfft, 50% overlap ');

figure();
spectrogram(x,blackman(512),[],512,fs,'yaxis');
title('window = hamming(512)');
