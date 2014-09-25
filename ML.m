function [dataReal,p0,p1]=ML(data,t,alter,K)
% [dataReal, p0, p1]=ML(data, t, alter, K)
% ML phase correction for temporal MRI data.
% 
% Takes a complex vector in time as input and makes it real by estimating 
% the linear phase by maximum likelihood. This avoids taking the magnitude, 
% which makes the noise distribution Rician. 
% 
% Inputs:
%           data  - complex-valued data vector over time
%           t     - time vector
%           alter - set = 1 for sign-alternating phases 
%           K     - Number of evaluated gridpoints in the FFT
%
% Outputs:
%           dataReal - ML estimated real-valued decay
%           p0       - constant coefficient
%           p1       - linear coefficient
%
% Marcus Björk 2014

%Make sure everything is in columns
data=data(:);
t=t(:);
if alter  %Conjugate every other sample
   data(2:2:end)=conj(data(2:2:end));
end
if nargin < 4
    K=1024; %Default grid size
end

if var(diff(t))<1e-10; %Uniform sampling -> FFT
    Ts=t(2)-t(1);
    [~,ind]=max(abs(fft(data.^2,K)));
    p1=(ind/K*pi-pi*(ind>K/2))/Ts; %Automatically incorporates the alternating phases by returning w>pi?
else %Nonuniform sampling -> gridsearch
    %There is no upper limit for the grid in the nonuniform case. But p1
    %should be small in general.
    p1_grid=linspace(-pi/(mean(diff(t))),pi/(mean(diff(t))),K).';
    %Nonuniform fourier matrix
    [~,ind]=max(abs(exp(-1i*2*p1_grid*t.')*data.^2));
    p1=p1_grid(ind);
end

%Solve for remaining parameters
p0=angle(exp(-1i*2*p1*t.')*data.^2)/2;
dataReal=real(exp(-1i*(p0+p1*t)).*data);

%To resolve sign ambiguity
if dataReal(1)<0
    dataReal=-dataReal;
    p0=p0+pi;
end
