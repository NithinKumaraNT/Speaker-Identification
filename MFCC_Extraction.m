%25-01-2018

%Speaker Recognition

% The following code extracts the feature vectors from an audio frame

%% Frame Extraction
     % TrainSamples >> Number of audio samples considered from Timit/Test

    % Input = An audio file *.wav
    %Requires the TestAudioDatabase from all the speakers in the test data
    clearvars -except MasterTestAudio means var weights;
    TotalSpeakers = 168;
    Data = [];
 for TrainSamples = 1 : TotalSpeakers*10
    Data = MasterTestAudio{TrainSamples,1}; %Acquire data from each audio file
    %sound(Data,16000); To play back the audio file
    TotalNumberOfSamples = size(Data,1);
    Fs = 16000;
    Tfeed = 10e-3; 
    Tframe = 20e-3;
    FrameLength = Fs * Tframe; % Length of each frame = 320 Samples 
    % Data =specsub(Data,Fs); % For Pre Emphasis if needed
    LastSample = int32(((TotalNumberOfSamples - FrameLength)/(Fs * Tfeed))+ 1); % Total number of frames
    AudioFrames = zeros(FrameLength,LastSample);
    for K = 0 : LastSample - 1
    for Frame = 1 : FrameLength
        if ((K * Fs* Tfeed) + Frame) <= TotalNumberOfSamples
       AudioFrames(Frame,K+1) = Data(((K * Fs* Tfeed) + Frame),1);  
        end
    end
    end
    clear K;
    % Output = Matrix with 320samples x NoOfFrames
    
     %% Voice activity detection

    %Step1 : Individual Signal power for all frames
    for K = 1 : LastSample
        SignalPower = 0;
    for i = 1 : FrameLength
       SignalPower = SignalPower + (AudioFrames(i,K) * AudioFrames(i,K));
    end
       % The AudioFrames matrix is concatenated with a new row with Signal power
       AudioFrames(FrameLength + 1 ,K) = SignalPower / FrameLength;
    end
    clear SignalPower;
    %Step2 : Calculate the noise power 
    Tnoise = 100e-3; %In seconds
    Knoise = ((Tnoise/Tfeed) - 1);
    NoisePower = 0;
    for K =1 : Knoise
        NoisePower = NoisePower + AudioFrames(FrameLength + 1,K);
    end
    Pnoise = NoisePower / Knoise ;
    clear NoisePower K;
    %Step3 : Obtain the Voiced and Unvoiced frames
    Gamma = 500;
    eliminate = [];
    for K = 1  : LastSample
        if AudioFrames(FrameLength + 1 ,K) < Gamma * Pnoise
            AudioFrames(FrameLength + 2 ,K) = 0;
            eliminate = [eliminate K];
        else
            AudioFrames(FrameLength + 2 ,K) = 1;
        end

    end
    UnVoicedFrames = AudioFrames(1:320 ,eliminate);
    AudioFrames(: ,eliminate) = [];
    VoicedFrames = AudioFrames(1:320,:);
    %sound(VoicedFrames(:),16000); % To playback the voiced frames
    
    
    %% Window Function
    %Pass each frame through a hamming window before fft
    VoicedFrames = VoicedFrames.*hamming(FrameLength);
    
    %% Mel Filter Bank
    b = zeros(1,15);
    MFCC = zeros(size(VoicedFrames,2),15);      % Pre allocate memory
      MFCC_test = zeros(size(VoicedFrames,2),15);
    MelBank=full(melbankm(22,320,16000,0,0.5));  % melbankm(NoOfFilters,fftlength,Fs,0,0.5); 0.5 -> Fs/2
    for Frame = 1 : size(VoicedFrames,2)
        f=rfft(VoicedFrames(:,Frame));               % rfft() returns only 1+floor(n/2) coefficients
        z=log10((MelBank/2)*abs(f));              % multiply x by the power spectrum
        %MFCC(Frame,1:15) = dct(z,15)';    
        for j = 1: 15
            sum = 0;
            for i = 1 : 22
                sum = sum + (z(i) *cos (((pi * j)/22)*(i - (1/2))));
            end
            b(1,j) = sum;
        end
        MFCC_test(Frame,1:15) = b;
    end
    MasterTestAudio{TrainSamples,6} = MFCC_test;
 end  
 
 function [x,mc,mn,mx]=melbankm(p,n,fs,fl,fh,w)
%MELBANKM determine matrix for a mel/erb/bark-spaced filterbank [X,MN,MX]=(P,N,FS,FL,FH,W)
%
% Inputs:
%       p   number of filters in filterbank or the filter spacing in k-mel/bark/erb [ceil(4.6*log10(fs))]
%		n   length of fft
%		fs  sample rate in Hz
%		fl  low end of the lowest filter as a fraction of fs [default = 0]
%		fh  high end of highest filter as a fraction of fs [default = 0.5]
%		w   any sensible combination of the following:
%             'b' = bark scale instead of mel
%             'e' = erb-rate scale
%             'l' = log10 Hz frequency scale

%


% Note "FFT bin_0" assumes DC = bin 0 whereas "FFT bin_1" means DC = bin 1

if nargin < 6
    w='tz'; % default options
end
if nargin < 5 || isempty(fh)
    fh=0.5; % max freq is the nyquist
end
if nargin < 4 || isempty(fl)
    fl=0; % min freq is DC
end

sfact=2-any(w=='s');   % 1 if single sided else 2
wr=' ';   % default warping is mel
for i=1:length(w)
    if any(w(i)=='lebf');
        wr=w(i);
    end
end
if any(w=='h') || any(w=='H')
    mflh=[fl fh];
else
    mflh=[fl fh]*fs;
end
if ~any(w=='H')
    switch wr
                    case 'f'       % no transformation
        case 'l'
            if fl<=0
                error('Low frequency limit must be >0 for l option');
            end
            mflh=log10(mflh);       % convert frequency limits into log10 Hz
        case 'e'
            mflh=frq2erb(mflh);       % convert frequency limits into erb-rate
        case 'b'
            mflh=frq2bark(mflh);       % convert frequency limits into bark
        otherwise
            mflh=frq2mel(mflh);       % convert frequency limits into mel
    end
end
melrng=mflh*(-1:2:1)';          % mel range
fn2=floor(n/2);     % bin index of highest positive frequency (Nyquist if n is even)
if isempty(p)
    p=ceil(4.6*log10(fs));         % default number of filters
end
if any(w=='c')              % c option: specify fiter centres not edges
if p<1
    p=round(melrng/(p*1000))+1;
end
melinc=melrng/(p-1);
mflh=mflh+(-1:2:1)*melinc;
else
    if p<1
    p=round(melrng/(p*1000))-1;
end
melinc=melrng/(p+1);
end

%
% Calculate the FFT bins corresponding to [filter#1-low filter#1-mid filter#p-mid filter#p-high]
%
switch wr
        case 'f'
        blim=(mflh(1)+[0 1 p p+1]*melinc)*n/fs;
    case 'l'
        blim=10.^(mflh(1)+[0 1 p p+1]*melinc)*n/fs;
    case 'e'
        blim=erb2frq(mflh(1)+[0 1 p p+1]*melinc)*n/fs;
    case 'b'
        blim=bark2frq(mflh(1)+[0 1 p p+1]*melinc)*n/fs;
    otherwise
        blim=mel2frq(mflh(1)+[0 1 p p+1]*melinc)*n/fs;
end
mc=mflh(1)+(1:p)*melinc;    % mel centre frequencies
b1=floor(blim(1))+1;            % lowest FFT bin_0 required might be negative)
b4=min(fn2,ceil(blim(4))-1);    % highest FFT bin_0 required
%
% now map all the useful FFT bins_0 to filter1 centres
%
switch wr
        case 'f'
        pf=((b1:b4)*fs/n-mflh(1))/melinc;
    case 'l'
        pf=(log10((b1:b4)*fs/n)-mflh(1))/melinc;
    case 'e'
        pf=(frq2erb((b1:b4)*fs/n)-mflh(1))/melinc;
    case 'b'
        pf=(frq2bark((b1:b4)*fs/n)-mflh(1))/melinc;
    otherwise
        pf=(frq2mel((b1:b4)*fs/n)-mflh(1))/melinc;
end
%
%  remove any incorrect entries in pf due to rounding errors
%
if pf(1)<0
    pf(1)=[];
    b1=b1+1;
end
if pf(end)>=p+1
    pf(end)=[];
    b4=b4-1;
end
fp=floor(pf);                  % FFT bin_0 i contributes to filters_1 fp(1+i-b1)+[0 1]
pm=pf-fp;                       % multiplier for upper filter
k2=find(fp>0,1);   % FFT bin_1 k2+b1 is the first to contribute to both upper and lower filters
k3=find(fp<p,1,'last');  % FFT bin_1 k3+b1 is the last to contribute to both upper and lower filters
k4=numel(fp); % FFT bin_1 k4+b1 is the last to contribute to any filters
if isempty(k2)
    k2=k4+1;
end
if isempty(k3)
    k3=0;
end
if any(w=='y')          % preserve power in FFT
    mn=1; % lowest fft bin required (1 = DC)
    mx=fn2+1; % highest fft bin required (1 = DC)
    r=[ones(1,k2+b1-1) 1+fp(k2:k3) fp(k2:k3) repmat(p,1,fn2-k3-b1+1)]; % filter number_1
    c=[1:k2+b1-1 k2+b1:k3+b1 k2+b1:k3+b1 k3+b1+1:fn2+1]; % FFT bin1
    v=[ones(1,k2+b1-1) pm(k2:k3) 1-pm(k2:k3) ones(1,fn2-k3-b1+1)];
else
    r=[1+fp(1:k3) fp(k2:k4)]; % filter number_1
    c=[1:k3 k2:k4]; % FFT bin_1 - b1
    v=[pm(1:k3) 1-pm(k2:k4)];
    mn=b1+1; % lowest fft bin_1
    mx=b4+1;  % highest fft bin_1
end
if b1<0
    c=abs(c+b1-1)-b1+1;     % convert negative frequencies into positive
end
% end
if any(w=='n')
    v=0.5-0.5*cos(v*pi);      % convert triangles to Hanning
elseif any(w=='m')
    v=0.5-0.46/1.08*cos(v*pi);  % convert triangles to Hamming
end
if sfact==2  % double all except the DC and Nyquist (if any) terms
    msk=(c+mn>2) & (c+mn<n-fn2+2);  % there is no Nyquist term if n is odd
    v(msk)=2*v(msk);
end
%
% sort out the output argument options
%
if nargout > 2
    x=sparse(r,c,v);
    if nargout == 3     % if exactly three output arguments, then
        mc=mn;          % delete mc output for legacy code compatibility
        mn=mx;
    end
else
    x=sparse(r,c+mn-1,v,p,1+fn2);
end
if any(w=='u')
    sx=sum(x,2);
    x=x./repmat(sx+(sx==0),1,size(x,2));
end
%
% plot results if no output arguments or g option given
%
if ~nargout || any(w=='g') % plot idealized filters
    ng=201;     % 201 points
    me=mflh(1)+(0:p+1)'*melinc;
    switch wr
                case 'f'
            fe=me; % defining frequencies
            xg=repmat(linspace(0,1,ng),p,1).*repmat(me(3:end)-me(1:end-2),1,ng)+repmat(me(1:end-2),1,ng);
        case 'l'
            fe=10.^me; % defining frequencies
            xg=10.^(repmat(linspace(0,1,ng),p,1).*repmat(me(3:end)-me(1:end-2),1,ng)+repmat(me(1:end-2),1,ng));
        case 'e'
            fe=erb2frq(me); % defining frequencies
            xg=erb2frq(repmat(linspace(0,1,ng),p,1).*repmat(me(3:end)-me(1:end-2),1,ng)+repmat(me(1:end-2),1,ng));
        case 'b'
            fe=bark2frq(me); % defining frequencies
            xg=bark2frq(repmat(linspace(0,1,ng),p,1).*repmat(me(3:end)-me(1:end-2),1,ng)+repmat(me(1:end-2),1,ng));
        otherwise
            fe=mel2frq(me); % defining frequencies
            xg=mel2frq(repmat(linspace(0,1,ng),p,1).*repmat(me(3:end)-me(1:end-2),1,ng)+repmat(me(1:end-2),1,ng));
    end

    v=1-abs(linspace(-1,1,ng));
    if any(w=='n')
        v=0.5-0.5*cos(v*pi);      % convert triangles to Hanning
    elseif any(w=='m')
        v=0.5-0.46/1.08*cos(v*pi);  % convert triangles to Hamming
    end
    v=v*sfact;  % multiply by 2 if double sided
    v=repmat(v,p,1);
    if any(w=='y')  % extend first and last filters
        v(1,xg(1,:)<fe(2))=sfact;
        v(end,xg(end,:)>fe(p+1))=sfact;
    end
    if any(w=='u') % scale to unity sum
        dx=(xg(:,3:end)-xg(:,1:end-2))/2;
        dx=dx(:,[1 1:ng-2 ng-2]);
        vs=sum(v.*dx,2);
        v=v./repmat(vs+(vs==0),1,ng)*fs/n;
    end
    plot(xg',v','b');
    set(gca,'xlim',[fe(1) fe(end)]);
    xlabel(['Frequency (' xticksi 'Hz)']);
end
 end
 
 
 
 
 function [mel,mr] = frq2mel(frq)
%FRQ2ERB  Convert Hertz to Mel frequency scale MEL=(FRQ)
%	[mel,mr] = frq2mel(frq) converts a vector of frequencies (in Hz)

persistent k
if isempty(k)
    k=1000/log(1+1000/700); %  1127.01048
end
af=abs(frq);
mel = sign(frq).*log(1+af/700)*k;
mr=(700+af)/k;
if ~nargout
    plot(frq,mel,'-x');
    xlabel(['Frequency (' xticksi 'Hz)']);
    ylabel(['Frequency (' yticksi 'Mel)']);
end
 end
 
 
 
 function [frq,mr] = mel2frq(mel)
%MEL2FRQ  Convert Mel frequency scale to Hertz FRQ=(MEL)
%	frq = mel2frq(mel) converts a vector of Mel frequencies
%	to the corresponding real frequencies.
%   mr gives the corresponding gradients in Hz/mel.
%	The Mel scale corresponds to the perceived pitch of a tone

%	The relationship between mel and frq is given by [1]:
%
%	m = ln(1 + f/700) * 1000 / ln(1+1000/700)
%
%  	This means that m(1000) = 1000
%

persistent k
if isempty(k)
    k=1000/log(1+1000/700); % 1127.01048
end
frq=700*sign(mel).*(exp(abs(mel)/k)-1);
mr=(700+abs(frq))/k;
if ~nargout
    plot(mel,frq,'-x');
    xlabel(['Frequency (' xticksi 'Mel)']);
    ylabel(['Frequency (' yticksi 'Hz)']);
end
 end
 
 
 
 function y=rfft(x,n,d)
%RFFT     Calculate the DFT of real data Y=(X,N,D)
% Data is truncated/padded to length N if specified.
%   N even:	(N+2)/2 points are returned with
% 			the first and last being real
%   N odd:	(N+1)/2 points are returned with the
% 			first being real
% In all cases fix(1+N/2) points are returned
% D is the dimension along which to do the DFT



s=size(x);
if prod(s)==1
    y=x
else
    if nargin <3 || isempty(d)
        d=find(s>1,1);
        if nargin<2
            n=s(d);
        end
    end
    if isempty(n) 
        n=s(d);
    end
    y=fft(x,n,d);
    y=reshape(y,prod(s(1:d-1)),n,prod(s(d+1:end))); 
    s(d)=1+fix(n/2);
    y(:,s(d)+1:end,:)=[];
    y=reshape(y,s);
end
 end
