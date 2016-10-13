function [dataReal, dataRot, x, critVal]=WELPE_mat(data_mat,t,order,smoothW)
% function [dataReal, dataRot, p0, p1, critVal]=WELPE_mat(data,t,smoothW)
% Phase correction for temporal MRI data.
% 
% Takes as input a complex-valued 3D matrix where the first two are spatial 
% dimensions and the 3rd dimension is time, and makes it real by estimating 
% the temporal phase function and projecting the signal onto the real axis. 
% This avoids taking the magnitude, which makes the noise distribution Rician. 
% This verions uses polynomials as basis functions only. But any set of basis 
% functions is possible. 
%
% Inputs:
%           data_mat - complex-valued data matrix over 2D space and time
%           t        - time vector
%           order    - order of the polynolial to fit
%           smoothW  - weight for the surrounding voxels in the LS fit
%
% Outputs:
%           dataReal - estimated real-valued decay of center voxel
%           dataRot  - phase-corrected data (complex-valued) of center voxel
%           x        - estimated linear parameters in the fit
%           critVal  - least squares phase fit criterion value
%
% Marcus Björk 2015

t=t(:);

%Compute size of the smoothing region
[Nrow, Ncol, N]=size(data_mat);

%Construct weighting matrix for surrounding voxels
smoothW_mat=smoothW*ones(N,Nrow,Ncol);
smoothW_mat(:,(Nrow+1)/2,(Ncol+1)/2)=1; %Center is unweighted
smoothW_vec=smoothW_mat(:);

%Reshape the 3D matrix data to column vector
data=reshape(permute(data_mat,[3 1 2]),[],1,1);

tol=0.9*2*pi; %Wrapping tolerance (could be changed)

%Linear regressor matrix
R=bsxfun(@power,t,0:order);
A=repmat(R,Nrow*Ncol,1); %Repeated matrix to cover several datasets
w=abs(data); %These are the sqrt(Weights), only used for implementation, the result is abs(data)^2
dataPhaseWeighted=unwrap1D_mat(angle(data),Nrow*Ncol,tol).*w.*smoothW_vec;
x=bsxfun(@times,w.*smoothW_vec,A)\dataPhaseWeighted;
%Correct the center voxel only
dataRot=squeeze(data_mat((Nrow+1)/2,(Ncol+1)/2,:)).*exp(-1i*R*x);

if nargout==4
    critVal=norm(unwrap1D_mat(angle(data),1,tol)-A*x);
end

%Return real part
dataReal=real(dataRot);

end

function p_full = unwrap1D_mat(p_full,Ndata,tol)
%UNWRAP1D_mat Unwrap phase angle. Simplified version of MATLABs
%implementation that supports unwrapping multiple parts of a vector
%individually, for example, if data is stacked in a long vector. Ndata is
%the number of datasets stacked.
%   UNWRAP1D(P) unwraps radian phases P by changing absolute 
%   jumps greater than or equal to tol to their 2*pi complement.

% Unwrap phase angles.  Algorithm minimizes the incremental phase variation 
% by constraining it to the range [-pi,pi]

p_full=p_full(:);

N=length(p_full)/Ndata; %Number of samples in each original data vector
for k=1:Ndata
    p=p_full(1+(k-1)*N:N*k); %Take out the part to unwrap
    dp = diff(p,1,1);                % Incremental phase variations
    dps = mod(dp+pi,2*pi) - pi;      % Equivalent phase variations in [-pi,pi)
    dps(dps==-pi & dp>0) = pi;     % Preserve variation sign for pi vs. -pi
    dp_corr = dps - dp;              % Incremental phase corrections
    dp_corr(abs(dp)<tol) = 0;   % Ignore correction when incr. variation is < CUTOFF

    % Integrate corrections and add to P to produce smoothed phase values
    p(2:end) = p(2:end) + cumsum(dp_corr,1);
    p_full(1+(k-1)*N:N*k)=p; %Write the result back to p_full
end
end