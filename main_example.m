%% 1. Simulated object
clear; close all; clc

% System parameters
N = 3; % Number of holograms
z = 2000; % Defocus distance of first hologram [um]
dz = 1000; % Defocus distance difference between subsequent holograms
dx = 2; % Effective pixel size [um]
lambda = 0.5; % Wavelength [um]
S = 600; % Image size [px]
noise_std = 0.1; % Noise standard deviation in the input holograms

% Measured object
usaf = USAFmask(S,dx,10,4:9,1);
usaf(usaf<0) = 0; usaf(usaf>1) = 1;
amp = 1-0.5*usaf;
phs = usaf;
% obj = amp; % amplitude type object
obj = exp(1i.*phs); % Phase type object

% Hologram generation
zz = z:dz:(z+(N-1)*dz);
H = zeros(S,S,N);
for nn = 1:N
    U = AS_propagate_p(obj,zz(nn),1,lambda,dx);
    H(:,:,nn) = (abs(U) + randn(S)*noise_std).^2;
end

% Hologram reconstruction
R_GHR = AS_propagate_p(H(:,:,1),-zz(1),1,lambda,dx);
R_GS = IGA(H,zz,lambda,dx,0,15);
R_IGA = IGA(H,zz,lambda,dx,1,15);
R_GA = IGA(H,zz,lambda,dx,inf,15);

% Displaying
if ~min(obj(:)==amp(:)) % phase object
    rng = [-1,1];
    ax = [];
    figure('Units','normalized','OuterPosition',[0,0.05,1,0.95])
    imagesc(angle(R_GHR),rng); colormap gray; axis image; ax = [ax,gca]; title('GHR')
    figure('Units','normalized','OuterPosition',[0,0.05,1,0.95])
    imagesc(angle(R_GS),rng); colormap gray; axis image; ax = [ax,gca]; title('GS')
    figure('Units','normalized','OuterPosition',[0,0.05,1,0.95])
    imagesc(angle(R_IGA),rng); colormap gray; axis image; ax = [ax,gca]; title('IGA (proposed)')
    figure('Units','normalized','OuterPosition',[0,0.05,1,0.95])
    imagesc(angle(R_GA),rng); colormap gray; axis image; ax = [ax,gca]; title('GA')
    figure('Units','normalized','OuterPosition',[0,0.05,1,0.95])
    imagesc(angle(obj),rng); colormap gray; axis image; ax = [ax,gca]; title('Simulated object')
    linkaxes(ax)
else % amplitude object
    rng = [0.2,1.3];
    ax = [];
    figure('Units','normalized','OuterPosition',[0,0.05,1,0.95])
    imagesc(abs(R_GHR),rng); colormap gray; axis image; ax = [ax,gca]; title('GHR')
    figure('Units','normalized','OuterPosition',[0,0.05,1,0.95])
    imagesc(abs(R_GS),rng); colormap gray; axis image; ax = [ax,gca]; title('GS')
    figure('Units','normalized','OuterPosition',[0,0.05,1,0.95])
    imagesc(abs(R_IGA),rng); colormap gray; axis image; ax = [ax,gca]; title('IGA (proposed)')
    figure('Units','normalized','OuterPosition',[0,0.05,1,0.95])
    imagesc(abs(R_GA),rng); colormap gray; axis image; ax = [ax,gca]; title('GA')
    figure('Units','normalized','OuterPosition',[0,0.05,1,0.95])
    imagesc(abs(obj),rng); colormap gray; axis image; ax = [ax,gca]; title('Simulated object')
    linkaxes(ax)
end

%% 2. Experimental data - QCI logo (TPP printed phase object)
clear; close all; clc

% Parameters
zz = [3108,4117,5120];
lambda = 0.561; 
dx = 2.4;

% Load data
load('Data\QCIlogoData.mat')

% Rescale and shift holograms
HPB = double(HPB); LPB1 = double(LPB1); LPB2 = double(LPB2);
Roriginal = imref2d(size(HPB(:,:,1)));
for tt = 1:2
    HPB(:,:,tt) = imwarp(HPB(:,:,tt),HPB_tform{tt},'OutputView',Roriginal);
    LPB1(:,:,tt) = imwarp(LPB1(:,:,tt),LPB1_tform{tt},'OutputView',Roriginal);
    LPB2(:,:,tt) = imwarp(LPB2(:,:,tt),LPB2_tform{tt},'OutputView',Roriginal);
end

% Transfer to GPU for faster processing
HPB = gpuArray(HPB);
LPB1 = gpuArray(LPB1);
LPB2 = gpuArray(LPB2);

% Reconstruct - HPB
R_GHR_HPB = AS_propagate_p(HPB(:,:,1),-zz(1),1,lambda,dx);
R_GS_HPB = IGA(HPB,zz,lambda,dx,0,5);
R_IGA_HPB = IGA(HPB,zz,lambda,dx,2,5);
R_GA_HPB = IGA(HPB,zz,lambda,dx,inf,5);

% Transfer back to CPU (to save memory)
R_GHR_HPB = gather(R_GHR_HPB);
R_GS_HPB = gather(R_GS_HPB);
R_IGA_HPB = gather(R_IGA_HPB);
R_GA_HPB = gather(R_GA_HPB);

% Display HPB
rng = [-1,1];
ax = [];
figure('Units','normalized','OuterPosition',[0,0.05,1,0.95])
imagesc(angle(R_GHR_HPB),rng); colormap gray; axis image; ax = [ax,gca]; title('GHR; HPB')
figure('Units','normalized','OuterPosition',[0,0.05,1,0.95])
imagesc(angle(R_GS_HPB),rng); colormap gray; axis image; ax = [ax,gca]; title('GS; HPB')
figure('Units','normalized','OuterPosition',[0,0.05,1,0.95])
imagesc(angle(R_IGA_HPB),rng); colormap gray; axis image; ax = [ax,gca]; title('IGA (proposed); HPB')
figure('Units','normalized','OuterPosition',[0,0.05,1,0.95])
imagesc(angle(R_GA_HPB),rng); colormap gray; axis image; ax = [ax,gca]; title('GA; HPB')
linkaxes(ax)
xlim([2301,2600]); ylim([1851,2100])

% Reconstruct - LPB1
R_GHR_LPB1 = AS_propagate_p(LPB1(:,:,1),-zz(1),1,lambda,dx);
R_GS_LPB1 = IGA(LPB1,zz,lambda,dx,0,5);
R_IGA_LPB1 = IGA(LPB1,zz,lambda,dx,2,5);
R_GA_LPB1 = IGA(LPB1,zz,lambda,dx,inf,5);

% Transfer back to CPU (to save memory)
R_GHR_LPB1 = gather(R_GHR_LPB1);
R_GS_LPB1 = gather(R_GS_LPB1);
R_IGA_LPB1 = gather(R_IGA_LPB1);
R_GA_LPB1 = gather(R_GA_LPB1);

% Display LPB1
rng = [-1,1];
ax = [];
figure('Units','normalized','OuterPosition',[0,0.05,1,0.95])
imagesc(angle(R_GHR_LPB1),rng); colormap gray; axis image; ax = [ax,gca]; title('GHR; LPB1')
figure('Units','normalized','OuterPosition',[0,0.05,1,0.95])
imagesc(angle(R_GS_LPB1),rng); colormap gray; axis image; ax = [ax,gca]; title('GS; LPB1')
figure('Units','normalized','OuterPosition',[0,0.05,1,0.95])
imagesc(angle(R_IGA_LPB1),rng); colormap gray; axis image; ax = [ax,gca]; title('IGA (proposed); LPB1')
figure('Units','normalized','OuterPosition',[0,0.05,1,0.95])
imagesc(angle(R_GA_LPB1),rng); colormap gray; axis image; ax = [ax,gca]; title('GA; LPB1')
linkaxes(ax)
xlim([2301,2600]); ylim([1851,2100])

% Reconstruct - LPB2
R_GHR_LPB2 = AS_propagate_p(LPB2(:,:,1),-zz(1),1,lambda,dx);
R_GS_LPB2 = IGA(LPB2,zz,lambda,dx,0,5);
R_IGA_LPB2 = IGA(LPB2,zz,lambda,dx,2,5);
R_GA_LPB2 = IGA(LPB2,zz,lambda,dx,inf,5);

% Transfer back to CPU (to save memory)
R_GHR_LPB2 = gather(R_GHR_LPB2);
R_GS_LPB2 = gather(R_GS_LPB2);
R_IGA_LPB2 = gather(R_IGA_LPB2);
R_GA_LPB2 = gather(R_GA_LPB2);

% Display LPB2
rng = [-pi/2,pi/2];
ax = [];
figure('Units','normalized','OuterPosition',[0,0.05,1,0.95])
imagesc(angle(R_GHR_LPB2),rng); colormap gray; axis image; ax = [ax,gca]; title('GHR; LPB2')
figure('Units','normalized','OuterPosition',[0,0.05,1,0.95])
imagesc(angle(R_GS_LPB2),rng); colormap gray; axis image; ax = [ax,gca]; title('GS; LPB2')
figure('Units','normalized','OuterPosition',[0,0.05,1,0.95])
imagesc(angle(R_IGA_LPB2),rng); colormap gray; axis image; ax = [ax,gca]; title('IGA (proposed); LPB2')
figure('Units','normalized','OuterPosition',[0,0.05,1,0.95])
imagesc(angle(R_GA_LPB2),rng); colormap gray; axis image; ax = [ax,gca]; title('GA; LPB2')
linkaxes(ax)
xlim([2251,2550]); ylim([1851,2100])

%% 3. Experimental data - phase resolution target
clear; close all; clc

zz = [4501, 5516, 8530];
lambda = 0.561; 
dx = 2.4;

% Load data
load('Data\PhaseTestData.mat')

% Rescale and shift holograms
HPB = double(HPB);
Roriginal = imref2d(size(HPB(:,:,1)));
for tt = 1:2
    HPB(:,:,tt) = imwarp(HPB(:,:,tt),tform{tt},'OutputView',Roriginal);
end

% Transfer to GPU for faster processing
HPB = gpuArray(HPB);

% Reconstruct
R_GHR = AS_propagate_p(HPB(:,:,1),-zz(1),1,lambda,dx);
R_GS = IGA(HPB,zz,lambda,dx,0,5);
R_IGA = IGA(HPB,zz,lambda,dx,1,5);
R_GA = IGA(HPB,zz,lambda,dx,inf,5);

% Display
rng = [-0.1,0.1];
ax = [];
figure('Units','normalized','OuterPosition',[0,0.05,1,0.95])
imagesc(angle(R_GHR),rng); colormap gray; axis image; ax = [ax,gca]; title('GHR; HPB')
figure('Units','normalized','OuterPosition',[0,0.05,1,0.95])
imagesc(angle(R_GS),rng); colormap gray; axis image; ax = [ax,gca]; title('GS; HPB')
figure('Units','normalized','OuterPosition',[0,0.05,1,0.95])
imagesc(angle(R_IGA),rng); colormap gray; axis image; ax = [ax,gca]; title('IGA (proposed); HPB')
figure('Units','normalized','OuterPosition',[0,0.05,1,0.95])
imagesc(angle(R_GA),rng); colormap gray; axis image; ax = [ax,gca]; title('GA; HPB')
linkaxes(ax)

%% 4. Experimental data - sperm sample
clear; close all; clc

% Parameters
zz = [70.2, 71.1, 72.5];
lambda = [635,450,532]/1000;
dx = 5.5/20;


% Load data
load('Data\SpermData.mat')

% Transfer to GPU for faster processing
HPB = gpuArray(HPB);
LPB = gpuArray(LPB);

% Reconstruct - HPB
R_GHR_HPB = AS_propagate_p(HPB(:,:,1),-zz(1),1,lambda(1),dx);
R_GS_HPB = IGA(HPB,zz,lambda,dx,0,5);
R_IGA_HPB = IGA(HPB,zz,lambda,dx,2,5);
R_GA_HPB = IGA(HPB,zz,lambda,dx,inf,5);

% Display HPB
rng = [-0.5,0.5];
ax = [];
figure('Units','normalized','OuterPosition',[0,0.05,1,0.95])
imagesc(-angle(R_GHR_HPB),rng); colormap gray; axis image; ax = [ax,gca]; title('GHR; HPB')
figure('Units','normalized','OuterPosition',[0,0.05,1,0.95])
imagesc(-angle(R_GS_HPB),rng); colormap gray; axis image; ax = [ax,gca]; title('GS; HPB')
figure('Units','normalized','OuterPosition',[0,0.05,1,0.95])
imagesc(-angle(R_IGA_HPB),rng); colormap gray; axis image; ax = [ax,gca]; title('IGA (proposed); HPB')
figure('Units','normalized','OuterPosition',[0,0.05,1,0.95])
imagesc(-angle(R_GA_HPB),rng); colormap gray; axis image; ax = [ax,gca]; title('GA; HPB')
linkaxes(ax)

% Reconstruct - LPB
R_GHR_LPB = AS_propagate_p(LPB(:,:,1),-zz(1),1,lambda(1),dx);
R_GS_LPB = IGA(LPB,zz,lambda,dx,0,5);
R_IGA_LPB = IGA(LPB,zz,lambda,dx,2,5);
R_GA_LPB = IGA(LPB,zz,lambda,dx,inf,5);

% Display LPB1
rng = [-0.5,0.5];
ax = [];
figure('Units','normalized','OuterPosition',[0,0.05,1,0.95])
imagesc(-angle(R_GHR_LPB),rng); colormap gray; axis image; ax = [ax,gca]; title('GHR; LPB')
figure('Units','normalized','OuterPosition',[0,0.05,1,0.95])
imagesc(-angle(R_GS_LPB),rng); colormap gray; axis image; ax = [ax,gca]; title('GS; LPB')
figure('Units','normalized','OuterPosition',[0,0.05,1,0.95])
imagesc(-angle(R_IGA_LPB),rng); colormap gray; axis image; ax = [ax,gca]; title('IGA (proposed); LPB')
figure('Units','normalized','OuterPosition',[0,0.05,1,0.95])
imagesc(-angle(R_GA_LPB),rng); colormap gray; axis image; ax = [ax,gca]; title('GA; LPB')
linkaxes(ax)
