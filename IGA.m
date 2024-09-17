function R = IGA(H,z,lambda,dx,sigma,iter)
% Iterative Gabor Averagin (IGA) - method for phase retrieval from multiple
% in-line holograms collected with different defocus and/or wavelength.
% Algorithm was designed to work with low signal-to-noise-ratio data
%
% Inputs:
%   H - 3D array containing registered in-line holograms (at least 2
%       hologram are reguired)
%   z - vector containing defocus distances for each hologram 
%       (AS_propagate_p(H(:,:,n),z(n),lambda(n),dx) should give the
%       in-focus reconstruction)
%   lambda - vector containing wavelengths used to collect each in-line
%       hologram
%   dx - effective pixel size of the camera (camera pixel size divided by 
%       system magnification)
%   sigma - denoising factor that balances the twin image and shot noise
%       removal. For larger sigma, shot noise should be minimized more
%       effectively, while twin image is minimized better for smaller
%       sigma. Default - sigma = 2; 
%       sigma = 0 - GS method. sigma = inf - GA method. 
%       Recommended values:
%       sigma = 0 - noise-free data (only simulations)
%       sigma = 1 - good quality data with insignificant shot noise
%       sigma = 2 - regular or noisy data
%       sigma = 4 - very strong shot noise, but twin image still present
%       sigma = inf - shot noise larger than signal
%   iter - number of iterations. Default - iter = 5;
% Output:
%   R - reconstructed complex optical field at the sample plane. 
%       abs(R) -> amplitude; angle(R) -> phase.
% 
% More details at/cite as:
%   M. Rogalski, P. Arcab, E. Wdowiak, J. Á. Picazo-Bueno, V. Micó, 
%   M. Józwik, M. Trusiak, "Hybrid iterating-averaging low photon budget 
%   Gabor holographic microscopy", Submitted 2024
% 
% Created by:
%   Mikołaj Rogalski
%   Warsaw University of Technology, Institute of Micromechanics and
%   Photonics
%   mikolaj.rogalski.dokt@pw.edu.pl
% 
% Last modified:
%   04.09.2024

[Sy,Sx,N] = size(H);

if sigma < inf % if not GA method
    if sigma > 0
        % Low-pass filtering of input holograms
        Hblurred = zeros(Sy,Sx,N);
        for n = 1:N
            Hblurred(:,:,n) = imgaussfilt(H(:,:,n),sigma);
        end
    else
        Hblurred = H; % No bluring for GS method
    end
    % Gerchberg-Saxton reconstruction
    if length(lambda) == 1
        R_GS = GS_multiHeight(Hblurred,z,lambda,dx,iter,N);
    else
        R_GS = GS_multiWavelength(Hblurred,z,lambda,dx,iter,N);
    end
end

% Gabor averaging reconstruction
if sigma > 0 % if not the GS method
    R_GA = GA(H,z,lambda,dx,N);
end

if sigma == 0 % GS method
    R = R_GS;
elseif sigma == inf % GA method
    R = R_GA;
else % IGA method
    % Combine GS and GA methods
    A = abs(R_GS) + abs(R_GA) - imgaussfilt(abs(R_GA),sigma);
    phi = angle(R_GS) + angle(R_GA) - imgaussfilt(angle(R_GA),sigma);
    R = A.*exp(1i.*phi);
end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function R_GS = GS_multiHeight(H,z,lambda,dx,iter,N)
% Multi-height Gerchberg-Saxton in-line holography phaes retrieval
    A = sqrt(H); % Amplitude = sqrt(intensity)
    U = A(:,:,1); % Initial guess of optical field in hologram 1 plane
    for tt = 1:iter
        for nn = 1:(N-1)
            % Propagate to nn+1 hologram plane
            U = AS_propagate_p(U,z(nn+1)-z(nn),lambda,dx);
            % Actualize optical field with nn+1 amplitude
            U = U./abs(U).*A(:,:,nn+1);
        end
        % Propagate to 1st hologram plane
        U = AS_propagate_p(U,z(1)-z(N),lambda,dx);
        % Actualize optical field with 1st amplitude
        U = U./abs(U).*A(:,:,1);
    end
    % Backpropagate reconstructed optical field to object plane
    R_GS = AS_propagate_p(U,-z(1),lambda,dx);
end

function R_GS = GS_multiWavelength(H,z,lambda,dx,iter,N)
% Multi-wavelength Gerchberg-Saxton in-line holography phaes retrieval
    A = sqrt(H); % Amplitude = sqrt(intensity)
    U = A(:,:,1); % Initial guess of optical field in hologram 1 plane
    for tt = 1:iter
        for nn = 1:(N-1)
            % Propagate to object plane
            R = AS_propagate_p(U,-z(nn),lambda(nn),dx);
            % Rescale phase to nn+1 wavelength
            phs = angle(R)*lambda(nn)/lambda(nn+1);
            R = abs(R).*exp(1i.*phs);
            % Propagate to nn+1 hologram plane
            U = AS_propagate_p(R,z(nn+1),lambda(nn+1),dx);
            % Actualize optical field with nn+1 amplitude
            U = U./abs(U).*A(:,:,nn+1);
        end
        % Propagate to object plane
        R = AS_propagate_p(U,-z(N),lambda(N),dx);
        % Rescale phase to 1st wavelength
        phs = angle(R)*lambda(N)/lambda(1);
        R = abs(R).*exp(1i.*phs);
        % Propagate to 1st hologram plane
        U = AS_propagate_p(R,z(1),lambda(1),dx);
        % Actualize optical field with 1st amplitude
        U = U./abs(U).*A(:,:,1);
    end
    % Backpropagate reconstructed optical field to object plane
    R_GS = AS_propagate_p(U,-z(1),lambda(1),dx);
end

function R_GA = GA(H,z,lambda,dx,N)
% Gabor averaging in-line holography reconstruction
    R_GA = 0;
    if length(lambda) == 1; lambda = zeros(1,N)+lambda; end
    for nn = 1:N
        % Backpropagate each hologram to object plane
        U = AS_propagate_p(H(:,:,nn),-z(nn),lambda(nn),dx);
        % Add the propagation result to the R_GA (and rescale phase to
        % match the 1st wavelength)
        R_GA = R_GA + sqrt(abs(U)).*exp(1i.*angle(U)*lambda(nn)/lambda(1));
    end
    % Divide by the number of holograms
    R_GA = R_GA/N;
end

function uout = AS_propagate_p(uin, z, lambda, dx)
% Angular spectrum optical field propagation
n0 = 1; % Medium refractive index
[Ny,Nx] = size(uin); % Image size
k = 2*pi/lambda; % Wavenumber

% Spatial frequencies
dfx = 1/Nx/dx; fx = -Nx/2*dfx : dfx : (Nx/2-1)*dfx; 
dfy = 1/Ny/dx; fy = -Ny/2*dfy : dfy : (Ny/2-1)*dfy; 

if  z<0 
    % Propagation kernel
    p = fftshift(k*z*sqrt(n0^2 - lambda^2*(ones(Ny,1)*(fx.^2)+(fy'.^2)*ones(1,Nx))));
    p = p - p(1,1); % Correct kernel so that propagated phase will have background = 0
    kernel = exp(-1i*p);
    % Propagate
    ftu = kernel.*fft2(conj(uin));
    uout = conj(ifft2(ftu));
else
    % Propagation kernel
    p = fftshift(k*z*sqrt(n0^2 - lambda^2*(ones(Ny,1)*(fx.^2)+(fy'.^2)*ones(1,Nx))));
    p = p - p(1,1); % Correct kernel so that propagated phase will have background = 0
    kernel = exp(1i*p);
    % Propagate
    ftu = kernel.*fft2(uin);
    uout = ifft2(ftu);
end
end