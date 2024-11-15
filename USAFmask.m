function [I,U] = USAFmask(S,dx,upSmpl,E,no)
% Function that creates a synthetic USAF target
%   Inputs:
%       S = [Sy,Sx]; or S = Sx; -> Image size [px].
%       dx -> Effective pixel size [um]
%       upSmpl -> Upsampling factor. USAF target will be generated as
%           binary mask with pixel size = dx/upSmpl and then downsampled to
%           dx pixel size
%       E -> USAF groups that will be genetated. Default E = 4:9;
%       no -> Add group and element numbers? 1-yes (default), 0-no
%   Outputs:
%       I -> Image of the USAF test (with dx sampling)
%       U -> Binary USAF mask (with dx/upSmpl sampling)
% dx should be at least 2-3 times smaller than dx_I to achieve sufficent
% sampling in the image. Also dx should be at least 2-3 times smaller than
% width of smallest line of latest group (e.g., for group 9 smallest line
% has 0.55 um width, then dx should be maximally around 0.275 (then in U,
% smallest line will be sampled with 2 pixels)

% Assign default values to variables if empty/not given
if nargin<1; S = [600,600]; end
if nargin<2; dx = 2; end
if nargin<3; upSmpl = 20; end
if nargin<4; E = 4:9; end
if nargin<5; no = 1; end
if isempty(S); S = [1200,1200]; end
if isempty(dx); dx = 2; end
if isempty(upSmpl); upSmpl = 20; end
if isempty(E); E = 4:9; end
if isempty(no); no = 1; end
if length(S) == 1; S = [S,S]; end

dx_U = dx/upSmpl;
% USAF pix location in x,y [um]
xx = -S(2)/2*dx+dx_U:dx_U:S(2)/2*dx;
yy = -S(1)/2*dx+dx_U:dx_U:S(1)/2*dx;
Wy = length(yy); Wx = length(xx);

% Initialize U
U = zeros(Wy,Wx);

q = 1; % USAF value (0 - background, q - USAF)

% First group number should be even
me = min(E); if mod(me,2) == 1; me = me - 1; end
Sx = 0; Dx = 0;
for ee = me:max(E)
    if mod(ee,2) == 0
        % Calculate bar widths for elements in this end next group
        Eve_d = 500./2.^(ee+(0:5)/6);
        Odd_d = 500./2.^(ee+1+(0:5)/6);

        Dx = round(Sx*0.1/dx_U) + Dx; % Shift in x of this 2 groups
        % Size of this 2 groups in xy
        Sx = 19*Eve_d(2) + 14*Odd_d(1);
        Sy = sum(Eve_d(2:6))*7-2*Eve_d(6);


        Scg = 5*Eve_d(2)/20/dx_U;
        d1g = Eve_d(2);

        if max(E==ee)
            % Create even group elements

            % 1st element
            d = Eve_d(1);
            L = 5*d;

            % Horizontal bars
            x = find(xx<Sx/2 & xx>Sx/2-L); x = x + Dx;
            y1 = find(yy<Sy/2 & yy>Sy/2-d);
            U(y1,x) = q;
            y2 = find(yy<Sy/2-2*d & yy>Sy/2-3*d);
            U(y2,x) = q;
            y3 = find(yy<Sy/2-4*d & yy>Sy/2-5*d);
            U(y3,x) = q;

            % Element number
            if no == 1
                Sc = 3*d/20/dx_U;
                d1 = d;
                im = txt2im(1);
                im = imresize(im,Sc,'nearest');
                [Qy,Qx] = size(im);
                [~,x0] = min(abs(xx-(Sx/2+d1))); x0 = x0 + Dx;
                x = x0+1:x0+Qx;
                [~,y0] = min(abs(yy-(Sy/2-2.5*d))); y0 = y0-round(Qy/2);
                y = y0+1:y0+Qy;

                im(1:1-y(1),:) = []; im(:,1:1-x(1)) = [];
                im((end-(y(end)-Wy))+1:end,:) = [];
                im(:,(end-(x(end)-Wx))+1:end) = [];
                im(:,1:1-x(1)) = [];
                y(y<1) = []; x(x<1) = []; y(y>Wy) = []; x(x>Wx) = [];
                U(y,x) = 1-im;
            end

            % Vertical bars
            x1 = find(xx<Sx/2-L-2*d & xx>Sx/2-L-3*d); x1 = x1 + Dx;
            y = find(yy<Sy/2 & yy>Sy/2-L);
            U(y,x1) = q;
            x2 = find(xx<Sx/2-L-4*d & xx>Sx/2-L-5*d); x2 = x2 + Dx;
            U(y,x2) = q;
            x3 = find(xx<Sx/2-L-6*d & xx>Sx/2-L-7*d); x3 = x3 + Dx;
            U(y,x3) = q;

            % Rectangle
            d = Eve_d(2);
            L = 5*d;
            x = find(xx>-Sx/2+14*d & xx<-Sx/2+14*d+L); x = x + Dx;
            y = find(yy>-Sy/2 & yy<-Sy/2+L);
            U(y,x) = q;

            % Group number
            if no == 1

                im = txt2im(ee);
                im = imresize(im,Scg,'nearest');
                [Qy,Qx] = size(im);
                [~,x0] = min(abs(xx-(-Sx/2+4*d1g))); x0 = x0 + Dx;
                x = x0+1:x0+Qx;
                [~,y0] = min(abs(yy-(-Sy/2-d1g))); y0 = y0-round(Qy);
                y = y0+1:y0+Qy;

                im(1:1-y(1),:) = []; im(:,1:1-x(1)) = [];
                im((end-(y(end)-Wy))+1:end,:) = [];
                im(:,(end-(x(end)-Wx))+1:end) = [];
                im(:,1:1-x(1)) = [];
                y(y<1) = []; x(x<1) = []; y(y>Wy) = []; x(x>Wx) = [];
                U(y,x) = 1-im;
            end

            % Elements from 2 to 6
            Lb = 0;
            for uu = 2:6
                d = Eve_d(uu);
                L = 5*d;

                % Horizontal bars
                x = find(xx>-Sx/2 & xx<-Sx/2+L); x = x + Dx;
                y1 = find(yy>-Sy/2+Lb & yy<-Sy/2+d+Lb);
                U(y1,x) = q;
                y2 = find(yy>-Sy/2+2*d+Lb & yy<-Sy/2+3*d+Lb);
                U(y2,x) = q;
                y3 = find(yy>-Sy/2+4*d+Lb & yy<-Sy/2+5*d+Lb);
                U(y3,x) = q;

                % Element number
                if no == 1
                    im = txt2im(uu);
                    im = imresize(im,Sc,'nearest');
                    [Qy,Qx] = size(im);
                    [~,x0] = min(abs(xx-(-Sx/2-d1))); x0 = x0 + Dx;
                    x = x0+1-Qx:x0;
                    [~,y0] = min(abs(yy-(-Sy/2+2.5*d+Lb))); y0 = y0-round(Qy/2);
                    y = y0+1:y0+Qy;
                    im(1:1-y(1),:) = []; im(:,1:1-x(1)) = [];
                    im((Qy-y(end)+Wy+1):Qy,:) = [];
                    im(:,(Qx-x(end)+Wx+1):Qx) = [];
                    y(y<1) = []; x(x<1) = []; y(y>Wy) = []; x(x>Wx) = [];
                    U(y,x) = 1-im;
                end
                % Vertical bars
                x1 = find(xx>-Sx/2+L+2*d & xx<-Sx/2+L+3*d); x1 = x1 + Dx;
                y = find(yy>-Sy/2+Lb & yy<-Sy/2+L+Lb);
                U(y,x1) = q;
                x2 = find(xx>-Sx/2+L+4*d & xx<-Sx/2+L+5*d); x2 = x2 + Dx;
                U(y,x2) = q;
                x3 = find(xx>-Sx/2+L+6*d & xx<-Sx/2+L+7*d); x3 = x3 + Dx;
                U(y,x3) = q;
                Lb = Lb + L + 2*d;
            end
        end
    elseif mod(ee,2) == 1
        if max(E==ee)
            % Create odd group elements

            % Group number
            if no == 1

                im = txt2im(ee);
                im = imresize(im,Scg,'nearest');
                [Qy,Qx] = size(im);
                [~,x0] = min(abs(xx-(Sx/2-2*d1g))); x0 = x0 + Dx;
                x = x0+1-Qx:x0;
                [~,y0] = min(abs(yy-(-Sy/2-d1g))); y0 = y0-round(Qy);
                y = y0+1:y0+Qy;

                im(1:1-y(1),:) = []; im(:,1:1-x(1)) = [];
                im((end-(y(end)-Wy))+1:end,:) = [];
                im(:,(end-(x(end)-Wx))+1:end) = [];
                im(:,1:1-x(1)) = [];
                y(y<1) = []; x(x<1) = []; y(y>Wy) = []; x(x>Wx) = [];
                U(y,x) = 1-im;
            end

            % Elements from 1 to 6
            Lb = 0;
            for uu = 1:6
                d = Odd_d(uu);
                L = 5*d;

                % Horizontal bars
                x = find(xx<Sx/2 & xx>Sx/2-L); x = x + Dx;
                y1 = find(yy>-Sy/2+Lb & yy<-Sy/2+d+Lb);
                U(y1,x) = q;
                y2 = find(yy>-Sy/2+2*d+Lb & yy<-Sy/2+3*d+Lb);
                U(y2,x) = q;
                y3 = find(yy>-Sy/2+4*d+Lb & yy<-Sy/2+5*d+Lb);
                U(y3,x) = q;

                % Element number
                if no == 1
                    if uu == 1
                        Sc = 3*d/20/dx_U;
                        d1 = d;
                    end
                    im = txt2im(uu);
                    im = imresize(im,Sc,'nearest');
                    [Qy,Qx] = size(im);
                    [~,x0] = min(abs(xx-(Sx/2+d1))); x0 = x0 + Dx;
                    x = x0+1:x0+Qx;
                    [~,y0] = min(abs(yy-(-Sy/2+2.5*d+Lb))); y0 = y0-round(Qy/2);
                    y = y0+1:y0+Qy;

                    im(1:1-y(1),:) = []; im(:,1:1-x(1)) = [];
                    im((end-(y(end)-Wy))+1:end,:) = [];
                    im(:,(end-(x(end)-Wx))+1:end) = [];
                    im(:,1:1-x(1)) = [];
                    y(y<1) = []; x(x<1) = []; y(y>Wy) = []; x(x>Wx) = [];
                    U(y,x) = 1-im;
                end

                % Vertical bars
                x1 = find(xx<Sx/2-L-2*d & xx>Sx/2-L-3*d); x1 = x1 + Dx;
                y = find(yy>-Sy/2+Lb & yy<-Sy/2+L+Lb);
                U(y,x1) = q;
                x2 = find(xx<Sx/2-L-4*d & xx>Sx/2-L-5*d); x2 = x2 + Dx;
                U(y,x2) = q;
                x3 = find(xx<Sx/2-L-6*d & xx>Sx/2-L-7*d); x3 = x3 + Dx;
                U(y,x3) = q;
                Lb = Lb + L + 2*d;
            end
        end
    end
end
I = imresize(U,1/upSmpl,'bilinear');
end

% Create numbers
function imtext = txt2im(nm)

text = num2str(nm);
text=text+0;                    % converting string into Ascii-number array
laenge=length(text);
imtext=zeros(20,18*laenge);     % Preparing the resulting image
for i=1:laenge

    code=text(i);

    if code==45
        TxtIm=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1;
            1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1];
    elseif code==48
        TxtIm=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1;
            1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1;
            1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1;
            1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1;
            1,1,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,1;
            1,1,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,1;
            1,1,0,0,0,0,1,1,0,0,1,1,0,0,0,0,1,1;
            1,1,0,0,0,0,1,1,0,0,1,1,0,0,0,0,1,1;
            1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,1,1;
            1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,1,1;
            1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1;
            1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1;
            1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1;
            1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1;
            1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1;
            1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1];
    elseif code==49
        TxtIm=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1;
            1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1;
            1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,1,1;
            1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,1,1;
            1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1;
            1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1;
            1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1;
            1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1;
            1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1;
            1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1;
            1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1;
            1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1;
            1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1;
            1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1;
            1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1;
            1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1];
    elseif code==50
        TxtIm=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1;
            1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1;
            1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1;
            1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1;
            1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1;
            1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1;
            1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1;
            1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1;
            1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1;
            1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1;
            1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1;
            1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1;
            1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1;
            1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1];
    elseif code==51
        TxtIm=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1;
            1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1;
            1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1;
            1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1;
            1,1,1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,1;
            1,1,1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1;
            1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1;
            1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1;
            1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1;
            1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1];
    elseif code==52
        TxtIm=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1;
            1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1;
            1,1,1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,1;
            1,1,1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,1;
            1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1;
            1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1;
            1,1,1,1,0,0,0,0,1,1,0,0,0,0,1,1,1,1;
            1,1,1,1,0,0,0,0,1,1,0,0,0,0,1,1,1,1;
            1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1;
            1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1;
            1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1;
            1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1;
            1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1;
            1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1;
            1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1;
            1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1];
    elseif code==53
        TxtIm=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1;
            1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1;
            1,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1;
            1,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1;
            1,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1;
            1,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1;
            1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1;
            1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1;
            1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1;
            1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1;
            1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1;
            1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1];
    elseif code==54
        TxtIm=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1;
            1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1;
            1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1;
            1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1;
            1,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1;
            1,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1;
            1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1;
            1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1;
            1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1;
            1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1;
            1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1;
            1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1;
            1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1;
            1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1;
            1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1;
            1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1];
    elseif code==55
        TxtIm=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1;
            1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1;
            1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1;
            1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1;
            1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1;
            1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1;
            1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1;
            1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1;
            1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1;
            1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1;
            1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1;
            1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1;
            1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1;
            1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1];
    elseif code==56
        TxtIm=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1;
            1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1;
            1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1;
            1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1;
            1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1;
            1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1;
            1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1;
            1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1;
            1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1;
            1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1;
            1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1;
            1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1;
            1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1;
            1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1;
            1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1;
            1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1];
    elseif code==57
        TxtIm=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1;
            1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1;
            1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1;
            1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1;
            1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1;
            1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1;
            1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1;
            1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1;
            1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1;
            1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1;
            1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1;
            1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1;
            1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,1,1;
            1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1];
    end
    imtext(1:20,((i-1)*18+1):i*18)=TxtIm;
end
end