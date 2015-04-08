function [dataReal, dataRot, p0, p1, critVal]=WELPE(data,t,order,alter)
% function [dataReal, dataRot, p0, p1, critVal]=WELPE(data,t,order,alter)
% Phase correction for temporal MRI data.
% 
% Takes a complex vector in time as input and makes it real by estimating 
% the phase and projecting the signal onto the real axis. This avoids 
% taking the magnitude, which makes the noise distribution Rician. 
% This verions uses polynomials as basis functions only. But any basis is 
% possible. 
%
% Inputs:
%           data  - complex-valued data vector over time
%           t     - time vector
%           order - order of the polynomial
%           alter - set = 1 for sign-alternating phases 
%
% Outputs:
%           dataReal - estimated real-valued decay
%           dataRot  - phase-corrected data (complex-valued)
%           p0       - constant coefficient in polynomial
%           p1       - linear coefficient in polynomial
%           critVal  - least squares phase fit criterion value
%
% Marcus Björk 2014

data=data(:);
t=t(:);

if alter  %Conjugate every other sample
   data(2:2:end)=conj(data(2:2:end));
end

tol=0.9*2*pi; %Wrapping tolerance

%Linear regressor matrix
A=bsxfun(@power,t,0:order);
w=abs(data); %These are the sqrt(Weights), only used for implementation, the result is abs(data)^2
dataPhaseWeighted=unwrap1D(angle(data),tol).*w;
x=bsxfun(@times,w,A)\dataPhaseWeighted;
dataRot=data.*exp(-1i*A*x);

if nargout==5
    critVal=norm(unwrap(unwrap(angle(data),tol))-A*x);
end
%Linear parameters
p0=x(1);
p1=x(2);
%Return real part
dataReal=real(dataRot);
