clear
close all

BeeSizeThreashold=50;
BeeSizeThreashold2=50;
LEDflash=2000;
videoFrames=62000;
%      v.CurrentTime = 0.6;   % go to particular timepoint  at 30FPS

% start video 1
v = VideoReader('Top88.MP4');
imshow(readFrame(v));
rectSync = round(getrect);

% start video 2
v2 = VideoReader('Side26.MP4');
imshow(readFrame(v2));
rectSync2 = round(getrect);
close all

load('stereoParamWei2018.mat')      % camera config from stereocal
tic
% find trigger 1
pic2sync3=readFrame(v);
pic2sync=pic2sync3(:,:,2);
syncPic=sum(sum(pic2sync(rectSync(2):rectSync(2)+rectSync(4),rectSync(1):rectSync(1)+rectSync(3))));
for i=1:3000
    pic2sync3_1=readFrame(v);
    pic2sync_1=pic2sync3_1(:,:,2);
    syncPic_1=sum(sum(pic2sync_1(rectSync(2):rectSync(2)+rectSync(4),rectSync(1):rectSync(1)+rectSync(3))));
    if syncPic_1-syncPic > LEDflash
       % imshow(pic2sync(rectSync(2):rectSync(2)+rectSync(4),rectSync(1):rectSync(1)+rectSync(3)));
        break
    end
    syncPic=syncPic_1;
end
disp(toc)
disp('found trigger 1')
% find trigger 2
pic2sync3=readFrame(v2);
pic2sync=pic2sync3(:,:,2);
syncPic=sum(sum(pic2sync(rectSync2(2):rectSync2(2)+rectSync2(4),rectSync2(1):rectSync2(1)+rectSync2(3))));
for i=1:3000
    pic2sync3_1=readFrame(v2);
    pic2sync_1=pic2sync3_1(:,:,2);
    syncPic_1=sum(sum(pic2sync_1(rectSync2(2):rectSync2(2)+rectSync2(4),rectSync2(1):rectSync2(1)+rectSync2(3))));
    if syncPic_1-syncPic > LEDflash
       % imshow(pic2sync(rectSync2(2):rectSync2(2)+rectSync2(4),rectSync2(1):rectSync2(1)+rectSync2(3)));
        break
    end
    syncPic=syncPic_1;
end
disp(toc)
disp('found trigger 2')
% jump over LEDflash
for i=1:200
    readFrame(v);
    readFrame(v2);
end

%I1 = undistortImage(readFrame(v),stereoParamsWEI.CameraParameters1);
%I2 = undistortImage(readFrame(v2),stereoParamsWEI.CameraParameters2);

% behaviour ROI
imshow(readFrame(v));
%imshow(I1)
rectBehaviour = getrect;
rectBehaviour = round(rectBehaviour);
imshow(readFrame(v2));
%imshow(I2)
rectBehaviour2 = getrect;
rectBehaviour2 = round(rectBehaviour2);
close all
%   Background for substraction
BGvideo=uint8(zeros(720,1280,10));
j=1;
disp(toc)
disp('beginn background substraction')
for i=1:999
    readFrame(v);
    if mod(i,100) == 0
        currentFrame = readFrame(v);
        BGvideo(:,:,j) = currentFrame(:,:,2);
        j=j+1;
    end
end
BG=median(BGvideo(:,:,:),3);
%clear BGvideo

BGvideo2=uint8(zeros(720,1280,10));
j=1;
for i=1:999
    readFrame(v2);
    if mod(i,100) == 0
        currentFrame = readFrame(v2);
        BGvideo2(:,:,j) = currentFrame(:,:,2);
        j=j+1;
    end
end
BG2=median(BGvideo2(:,:,:),3);
%clear BGvideo
disp(toc)
disp('ended background substraction')
%{
 imshow(BG);
 figure
 imshow(BG-BGvideo(:,:,1));
  if threshold to noisy or no bees
for i=1:99
 level(i) = graythresh(BG(rect(2):rect(2)+rect(4),rect(1):rect(1)+rect(3))-BGvideo(rect(2):rect(2)+rect(4),rect(1):rect(1)+rect(3),i))
end
 will be commentet out later
 for i=1:99
 imshow(imbinarize(BG(rect(2):rect(2)+rect(4),rect(1):rect(1)+rect(3))-BGvideo(rect(2):rect(2)+rect(4),rect(1):rect(1)+rect(3),i)));
  drawnow
 end
%}

bees=zeros(6,20,videoFrames);
for i=1:videoFrames
    currentFrame = readFrame(v);
    if mod(i,10) == 0     
        BGvideo(:,:,rem(i,10)+1) = currentFrame(:,:,2);
        BG=median(BGvideo(:,:,:),3);
    end    
    BWcurrent = imbinarize(BG(rectBehaviour(2):rectBehaviour(2)+rectBehaviour(4), ...
        rectBehaviour(1):rectBehaviour(1)+rectBehaviour(3))-currentFrame(rectBehaviour(2): ...
        rectBehaviour(2)+rectBehaviour(4),rectBehaviour(1):rectBehaviour(1)+rectBehaviour(3),2));
    gray=imgaussfilt(uint8(BWcurrent)*255,2.5);
    BWcurrent=imbinarize(gray,0.05);
    if sum(sum(BWcurrent))<10000 % if less than 10% white not an empty frame
        tracksCurrent = regionprops(BWcurrent,'centroid','Area','Orientation');
        k=0;
        for j=1:size(tracksCurrent,1)
            if tracksCurrent(j).Area>BeeSizeThreashold 
                k=k+1;
                bees(1,k,i) = tracksCurrent(j).Centroid(1);
                bees(2,k,i) = tracksCurrent(j).Centroid(2);
                bees(3,k,i) = tracksCurrent(j).Area;
            end
        end
    end
end

for i=1:videoFrames
    currentFrame = readFrame(v2);
    if mod(i,10) == 0     
        BGvideo2(:,:,rem(i,10)+1) = currentFrame(:,:,2);
        BG=median(BGvideo2(:,:,:),3);
    end   
    BWcurrent = imbinarize(BG2(rectBehaviour2(2):rectBehaviour2(2)+rectBehaviour2(4), ...
        rectBehaviour2(1):rectBehaviour2(1)+rectBehaviour2(3))-currentFrame(rectBehaviour2(2): ...
        rectBehaviour2(2)+rectBehaviour2(4),rectBehaviour2(1):rectBehaviour2(1)+rectBehaviour2(3),2));
    gray=imgaussfilt(uint8(BWcurrent)*255,2.5);
    BWcurrent=imbinarize(gray,0.05);
    if sum(sum(BWcurrent))<10000 % if less than 10% white not an empty frame
        tracksCurrent = regionprops(BWcurrent,'centroid','Area','Orientation');
        k=0;
        for j=1:size(tracksCurrent,1)
            if tracksCurrent(j).Area>BeeSizeThreashold2 
                k=k+1;
                bees(4,k,i) = tracksCurrent(j).Centroid(1);
                bees(5,k,i) = tracksCurrent(j).Centroid(2);
                bees(6,k,i) = tracksCurrent(j).Area;
            end
        end
    end
end
disp(toc)
disp('ended tracking')

bees(bees==0)=nan;
%beesFull=bees;
%bees2=beesFull;


%% here the monteCarloCode for top

Y = cell(1,videoFrames);
X = cell(1,videoFrames);
for i=1:videoFrames
aX=squeeze(bees(1,:,i));  
aX(isnan(aX))=[];
X{i}=aX';
aY=squeeze(bees(2,:,i));  
aY(isnan(aY))=[];
Y{i}=aY';
end   

% define main variables for KALMAN FILTER! :P
dt = 1;  %our sampling rate
S_frame = 5; % find(cellfun(@length, X)>11,1); %starting frame

%now, since we have multiple flies, we need a way to deal with a changing
%number of estimates! this way seems more clear for a tutorial I think, but
%there is probably a much more efficient way to do it.

u = 0; % define acceleration magnitude to start
HexAccel_noise_mag = 1; %process noise: the variability in how fast the Hexbug is speeding up (stdv of acceleration: meters/sec^2)
tkn_x = .1;  %measurement noise in the horizontal direction (x axis).
tkn_y = .1;  %measurement noise in the horizontal direction (y axis).
Ez = [tkn_x 0; 0 tkn_y];
Ex = [dt^4/4 0 dt^3/2 0; ...
    0 dt^4/4 0 dt^3/2; ...
    dt^3/2 0 dt^2 0; ...
    0 dt^3/2 0 dt^2].*HexAccel_noise_mag^2; % Ex convert the process noise (stdv) into covariance matrix
P = Ex; % estimate of initial Hexbug position variance (covariance matrix)

% Define update equations in 2-D! (Coefficent matrices): A physics based model for where we expect the HEXBUG to be [state transition (state + velocity)] + [input control (acceleration)]
A = [1 0 dt 0; 0 1 0 dt; 0 0 1 0; 0 0 0 1]; %state update matrice
B = [(dt^2/2); (dt^2/2); dt; dt];
C = [1 0 0 0; 0 1 0 0];  %this is our measurement function C, that we apply to the state estimate Q to get our expect next/new measurement

% initize result variables
Q_loc_meas = []; % the fly detecions  extracted by the detection algo
% initize estimation variables for two dimensions
Q= [X{S_frame} Y{S_frame} zeros(length(X{S_frame}),1) zeros(length(X{S_frame}),1)]';
Q_estimate = nan(4,6000);
Q_estimate(:,1:size(Q,2)) = Q;  %estimate of initial location estimation of where the flies are(what we are updating)
Q_loc_estimateY = nan(6000); %  position estimate
Q_loc_estimateX= nan(6000); %  position estimate
P_estimate = P;  %covariance estimator
strk_trks = zeros(1,6000);  %counter of how many strikes a track has gotten
nD = size(X{S_frame},1); %initize number of detections
nF =  find(isnan(Q_estimate(1,:))==1,1)-1 ; %initize number of track estimates

%for each frame
for t = S_frame:videoFrames  %length(f_list)-1 
    
    % load the image
  %  img_tmp = double(imread(f_list(t).name));
   % img = img_tmp(:,:,1);
    % make the given detections matrix
    Q_loc_meas = [X{t} Y{t}];
    
    % do the kalman filter
    % Predict next state of the flies with the last state and predicted motion.
    nD = size(X{t},1); %set new number of detections
    for F = 1:nF
        Q_estimate(:,F) = A * Q_estimate(:,F) + B * u;
    end
    
    %predict next covariance
    P = A * P* A' + Ex;
    % Kalman Gain
    K = P*C'*inv(C*P*C'+Ez);
    
    % now we assign the detections to estimated track positions
    %make the distance (cost) matrice between all pairs rows = tracks, coln =
    %detections
    est_dist = pdist([Q_estimate(1:2,1:nF)'; Q_loc_meas]);
    est_dist = squareform(est_dist); %make square
    est_dist = est_dist(1:nF,nF+1:end) ; %limit to just the tracks to detection distances
    
    [asgn, cost] = assignmentoptimal(est_dist); %do the assignment with hungarian algo
    asgn = asgn';
   % asgn(isempty(asgn))=0;
    % ok, now we check for tough situations and if it's tough, just go with estimate and ignore the data
    %make asgn = 0 for that tracking element
    if  isempty(asgn)==false
    %check 1: is the detection far from the observation? if so, reject it.
    rej = [];
    for F = 1:nF
        if asgn(F) > 0
            rej(F) =  est_dist(F,asgn(F)) < 50 ;
        else
            rej(F) = 0;
        end
    end
    asgn = asgn.*rej;
    
    %apply the assingment to the update
    k = 1;
    for F = 1:length(asgn)
        if asgn(F) > 0
            Q_estimate(:,k) = Q_estimate(:,k) + K * (Q_loc_meas(asgn(F),:)' - C * Q_estimate(:,k));
        end
        k = k + 1;
    end
    
    % update covariance estimation.
    P =  (eye(4)-K*C)*P;
    
    % Store data
    Q_loc_estimateX(t,1:nF) = Q_estimate(1,1:nF);
    Q_loc_estimateY(t,1:nF) = Q_estimate(2,1:nF);
    
    %ok, now that we have our assignments and updates, lets find the new detections and
    %lost trackings
    
    %find the new detections. basically, anything that doesn't get assigned
    %is a new tracking
    new_trk = [];
    new_trk = Q_loc_meas(~ismember(1:size(Q_loc_meas,1),asgn),:)';
    if ~isempty(new_trk)
        Q_estimate(:,nF+1:nF+size(new_trk,2))=  [new_trk; zeros(2,size(new_trk,2))];
        nF = nF + size(new_trk,2);  % number of track estimates with new ones included
    end
    
    %give a strike to any tracking that didn't get matched up to a
    %detection
    no_trk_list =  find(asgn==0);
    if ~isempty(no_trk_list)
        strk_trks(no_trk_list) = strk_trks(no_trk_list) + 1;
    end
    
    %if a track has a strike greater than 6, delete the tracking. i.e.
    %make it nan first vid = 3
    bad_trks = find(strk_trks > 6);
    Q_estimate(:,bad_trks) = NaN;
    end
end
disp('done with 1. monteCarlo')


%% here the monteCarloCode for side

Y = cell(1,videoFrames);
X = cell(1,videoFrames);
for i=1:videoFrames
aX=squeeze(bees(4,:,i));  
aX(isnan(aX))=[];
X{i}=aX';
aY=squeeze(bees(5,:,i));  
aY(isnan(aY))=[];
Y{i}=aY';
end   

% define main variables for KALMAN FILTER! :P
dt = 1;  %our sampling rate
S_frame = 5; % find(cellfun(@length, X)>11,1); %starting frame

%now, since we have multiple flies, we need a way to deal with a changing
%number of estimates! this way seems more clear for a tutorial I think, but
%there is probably a much more efficient way to do it.

u = 0; % define acceleration magnitude to start
HexAccel_noise_mag = 1; %process noise: the variability in how fast the Hexbug is speeding up (stdv of acceleration: meters/sec^2)
tkn_x = .1;  %measurement noise in the horizontal direction (x axis).
tkn_y = .1;  %measurement noise in the horizontal direction (y axis).
Ez = [tkn_x 0; 0 tkn_y];
Ex = [dt^4/4 0 dt^3/2 0; ...
    0 dt^4/4 0 dt^3/2; ...
    dt^3/2 0 dt^2 0; ...
    0 dt^3/2 0 dt^2].*HexAccel_noise_mag^2; % Ex convert the process noise (stdv) into covariance matrix
P = Ex; % estimate of initial Hexbug position variance (covariance matrix)

% Define update equations in 2-D! (Coefficent matrices): A physics based model for where we expect the HEXBUG to be [state transition (state + velocity)] + [input control (acceleration)]
A = [1 0 dt 0; 0 1 0 dt; 0 0 1 0; 0 0 0 1]; %state update matrice
B = [(dt^2/2); (dt^2/2); dt; dt];
C = [1 0 0 0; 0 1 0 0];  %this is our measurement function C, that we apply to the state estimate Q to get our expect next/new measurement

% initize result variables
Q_loc_meas = []; % the fly detecions  extracted by the detection algo
% initize estimation variables for two dimensions
Q= [X{S_frame} Y{S_frame} zeros(length(X{S_frame}),1) zeros(length(X{S_frame}),1)]';
Q_estimate = nan(4,6000);
Q_estimate(:,1:size(Q,2)) = Q;  %estimate of initial location estimation of where the flies are(what we are updating)
Q_loc_estimateY2 = nan(6000); %  position estimate
Q_loc_estimateX1= nan(6000); %  position estimate
P_estimate = P;  %covariance estimator
strk_trks = zeros(1,6000);  %counter of how many strikes a track has gotten
nD = size(X{S_frame},1); %initize number of detections
nF =  find(isnan(Q_estimate(1,:))==1,1)-1 ; %initize number of track estimates

%for each frame
for t = S_frame:videoFrames  %length(f_list)-1 
    
    % load the image
  %  img_tmp = double(imread(f_list(t).name));
   % img = img_tmp(:,:,1);
    % make the given detections matrix
    Q_loc_meas = [X{t} Y{t}];
    % do the kalman filter
    % Predict next state of the flies with the last state and predicted motion.
    nD = size(X{t},1); %set new number of detections
    for F = 1:nF
        Q_estimate(:,F) = A * Q_estimate(:,F) + B * u;
    end
    %predict next covariance
    P = A * P* A' + Ex;
    % Kalman Gain
    K = P*C'*inv(C*P*C'+Ez);
    
    % now we assign the detections to estimated track positions
    %make the distance (cost) matrice between all pairs rows = tracks, coln =
    %detections
    est_dist = pdist([Q_estimate(1:2,1:nF)'; Q_loc_meas]);
    est_dist = squareform(est_dist); %make square
    est_dist = est_dist(1:nF,nF+1:end) ; %limit to just the tracks to detection distances
    
    [asgn, cost] = assignmentoptimal(est_dist); %do the assignment with hungarian algo
    asgn = asgn';
   % asgn(isempty(asgn))=0;
    % ok, now we check for tough situations and if it's tough, just go with estimate and ignore the data
    %make asgn = 0 for that tracking element
    if  isempty(asgn)==false
    %check 1: is the detection far from the observation? if so, reject it.
    rej = [];
    for F = 1:nF
        if asgn(F) > 0
            rej(F) =  est_dist(F,asgn(F)) < 50 ;
        else
            rej(F) = 0;
        end
    end
    asgn = asgn.*rej;
    
    %apply the assingment to the update
    k = 1;
    for F = 1:length(asgn)
        if asgn(F) > 0
            Q_estimate(:,k) = Q_estimate(:,k) + K * (Q_loc_meas(asgn(F),:)' - C * Q_estimate(:,k));
        end
        k = k + 1;
    end
    
    % update covariance estimation.
    P =  (eye(4)-K*C)*P;
    
    % Store data
    Q_loc_estimateX1(t,1:nF) = Q_estimate(1,1:nF);
    Q_loc_estimateY2(t,1:nF) = Q_estimate(2,1:nF);
    
    %ok, now that we have our assignments and updates, lets find the new detections and
    %lost trackings
    
    %find the new detections. basically, anything that doesn't get assigned
    %is a new tracking
    new_trk = [];
    new_trk = Q_loc_meas(~ismember(1:size(Q_loc_meas,1),asgn),:)';
    if ~isempty(new_trk)
        Q_estimate(:,nF+1:nF+size(new_trk,2))=  [new_trk; zeros(2,size(new_trk,2))];
        nF = nF + size(new_trk,2);  % number of track estimates with new ones included
    end
    
    %give a strike to any tracking that didn't get matched up to a
    %detection
    no_trk_list =  find(asgn==0);
    if ~isempty(no_trk_list)
        strk_trks(no_trk_list) = strk_trks(no_trk_list) + 1;
    end
    
    %if a track has a strike greater than 6, delete the tracking. i.e.
    %make it nan first vid = 3
    bad_trks = find(strk_trks > 6);
    Q_estimate(:,bad_trks) = NaN;
    end
end

disp('done with 2. monteCarlo')

%% undistort

  Xtop=(Q_loc_estimateX1+rectBehaviour(1))*4.167;
  Ytop=(Q_loc_estimateY2+rectBehaviour(2))*3.125;
  Xside=(Q_loc_estimateX+rectBehaviour(1))*4.167;
  Yside=(Q_loc_estimateY+rectBehaviour(2))*3.125;
%  Xtop(Xtop<1200)=nan;
%  Xtop(Xtop>3700)=nan;
%  Ytop(Ytop<200)=nan;
%  Ytop(Ytop>2200)=nan;
%  Xside(Xside<1250)=nan;
%  Xside(Xside>3350)=nan;
%  Yside(Yside<1100)=nan;
%  Yside(Yside>2200)=nan;

for i=1:size(Xtop,2)
[row1,  ] = find(isfinite(Xtop(:,i)));
[rowrow,  ]=find(diff(row1)>1);
blub=min(rowrow);
if isempty(rowrow)
    blub=max(row1);
else
    blub=blub+min(row1)-1;
end
DATA(i).Xtop=Xtop(min(row1):blub,i); 
DATA(i).Ytop=Ytop(min(row1):blub,i); 
DATA(i).starttop=min(row1);
DATA(i).durtop=1+blub-min(row1);
DATA(i).endtop=blub;

[row2,  ] = find(isfinite(Xside(:,i)));
[rowrow2,  ]=find(diff(row2)>1);
blub2=min(rowrow2);
if isempty(rowrow2)
    blub2=max(row2);
else
    blub2=blub2+min(row2)-1;
end
DATA(i).Xside=Xside(min(row2):blub2,i); 
DATA(i).Yside=Yside(min(row2):blub2,i); 
DATA(i).startside=min(row2);
DATA(i).durside=1+blub2-min(row2);
DATA(i).endside=blub2;
%if isempty(row1) && isempty(row2) 
%    break
%end
end

for i=1:length(DATA)
    if ~isempty(DATA(i).Xtop) && ~isempty(DATA(i).Xside)
[DATA(i).topU]=undistortPoints([DATA(i).Xtop,DATA(i).Ytop],stereoParams2018.CameraParameters1);
[DATA(i).sideU]=undistortPoints([DATA(i).Xside,DATA(i).Yside],stereoParams2018.CameraParameters2);
%i
    end
end


%% controll code
%{
figure
for i=1:length(DATA)
hold on
%plot(DATA(i).topU(:,1),DATA(i).topU(:,2))
%subplot(1,2,2)
%hold on
if DATA(i).topU(1,1)<1300 && DATA(i).topU(end,1)>3300
plot(DATA(i).topU(:,1),DATA(i).topU(:,2),'r')
outbound(i)=nanmean(diff(DATA(i).topU(:,1)));
end
if DATA(i).topU(1,1)>1300 && DATA(i).topU(end,1)<3300
plot(DATA(i).topU(:,1),DATA(i).topU(:,2),'b')
inbound(i)=nanmean(diff(DATA(i).topU(:,1)));
end
end

subplot(1,2,1)
outbound(outbound==0)=nan;
nanmean(outbound)
plot(outbound,'*')
title('outbound')
subplot(1,2,2)
inbound(inbound>-5)=nan;
inbound(inbound==0)=nan;
nanmean(inbound)
plot(inbound,'*')
title('inbound')
nanmean(outbound)/nanmean(inbound)
sum(isfinite(outbound))
sum(isfinite(inbound))
%}


%% less then 30 datapoints per track is too little
for i=1:length(DATA)
if DATA(i).durtop < 30
    DATA(i).durtop=nan;
    DATA(i).topU=nan;
    DATA(i).starttop=nan;
    DATA(i).endtop=nan;
end
if DATA(i).durside < 30
    DATA(i).durside=nan;
    DATA(i).sideU=nan;
    DATA(i).startside=nan;
    DATA(i).endside=nan;
end
end
DATA=rmfield(DATA,{'Xtop','Ytop','Xside','Yside'});

%% inbound vs outbound marking
for i=1:length(DATA)
if DATA(i).topU(1,1)<1300 && DATA(i).topU(end,1)>3300  % out
DATA(i).directiontop=1;
end
if DATA(i).topU(1,1)>1300 && DATA(i).topU(end,1)<3300  % in
DATA(i).directiontop=-1;
end
if DATA(i).sideU(1,1)<1300 && DATA(i).sideU(end,1)>3300  % out
DATA(i).directionside=1;
end
if DATA(i).sideU(1,1)>1300 && DATA(i).sideU(end,1)<3300  % in
DATA(i).directionside=-1;
end

end


%% hornet    ########################
figure
hold on
for i=1:length(DATA) 
if DATA(i).durtop > 500 && isfinite(DATA(i).durtop) && isempty(DATA(i).directiontop)
    plot(DATA(i).topU(:,1),DATA(i).topU(:,2),'r')
    DATA(i).directiontop=0;
end
if DATA(i).durside > 200 && isfinite(DATA(i).durside) && isempty(DATA(i).directionside)
    plot(DATA(i).sideU(:,1),DATA(i).sideU(:,2),'b')
    DATA(i).directionside=0;
end
end


%% hornet traj with DATA

n=1;
for i=1:length(DATA)
    if DATA(i).directiontop == 0
        if DATA(i).durtop > 20
            for j=1:length(DATA)
                if DATA(j).directionside == 0
                    if DATA(j).durside > 20
                        if (DATA(i).starttop > DATA(j).startside && DATA(i).starttop < DATA(j).endside) || (DATA(j).startside > DATA(i).starttop && DATA(j).startside < DATA(i).endtop)
                                   bothCams= [max([DATA(i).starttop DATA(j).startside])  min([DATA(i).endtop  DATA(j).endside])];
                                   hornet3D(n).track=triangulate(DATA(i).topU(abs(bothCams(1)-DATA(i).starttop+1):abs(bothCams(2)-DATA(i).starttop+1),:),DATA(j).sideU(abs(bothCams(1)-DATA(j).startside+1):abs(bothCams(2)-DATA(j).startside+1),:),stereoParams2018); 
                                   hornet3D(n).start=bothCams(1);
                                   hornet3D(n).end=bothCams(2);
                                   n=n+1;
                        end
                    end
                end
            end
        end
    end
end

figure
for i=1:length(hornet3D)
hold on
plot3(hornet3D(i).track(:,1),hornet3D(i).track(:,2),hornet3D(i).track(:,3))
end


%% trajec bees  OUTBOUND

n=1;
for i=1:length(DATA)
    if DATA(i).directiontop == 1
        if DATA(i).durtop > 10
            for j=1:length(DATA)
                if DATA(j).directionside == 1
                    if DATA(j).durside > 10
                        if (DATA(i).starttop > DATA(j).startside && DATA(i).starttop < DATA(j).endside) || (DATA(j).startside > DATA(i).starttop && DATA(j).startside < DATA(i).endtop)
                                   bothCams= [max([DATA(i).starttop DATA(j).startside])  min([DATA(i).endtop  DATA(j).endside])];
                                   outbees(n).track=triangulate(DATA(i).topU(abs(bothCams(1)-DATA(i).starttop+1):abs(bothCams(2)-DATA(i).starttop+1),:),DATA(j).sideU(abs(bothCams(1)-DATA(j).startside+1):abs(bothCams(2)-DATA(j).startside+1),:),stereoParams2018); 
                                   outbees(n).start=bothCams(1);
                                   outbees(n).end=bothCams(2);
                                   n=n+1;
                        end
                    end
                end
            end
        end
    end
end


figure
for i=1:length(outbees)
hold on
plot3(outbees(i).track(:,1),outbees(i).track(:,2),outbees(i).track(:,3))
end


%% trajec bees  INBOUND

n=1;
for i=1:length(DATA)
    if DATA(i).directiontop == -1
        if DATA(i).durtop > 10
            for j=1:length(DATA)
                if DATA(j).directionside == -1
                    if DATA(j).durside > 10
                        if (DATA(i).starttop > DATA(j).startside && DATA(i).starttop < DATA(j).endside) || (DATA(j).startside > DATA(i).starttop && DATA(j).startside < DATA(i).endtop)
                                   bothCams= [max([DATA(i).starttop DATA(j).startside])  min([DATA(i).endtop  DATA(j).endside])];
                                   inbees(n).track=triangulate(DATA(i).topU(abs(bothCams(1)-DATA(i).starttop+1):abs(bothCams(2)-DATA(i).starttop+1),:),DATA(j).sideU(abs(bothCams(1)-DATA(j).startside+1):abs(bothCams(2)-DATA(j).startside+1),:),stereoParams2018); 
                                   inbees(n).start=bothCams(1);
                                   inbees(n).end=bothCams(2);
                                   n=n+1;
                        end
                    end
                end
            end
        end
    end
end


figure
for i=1:length(inbees)
hold on
plot3(inbees(i).track(:,1),inbees(i).track(:,2),inbees(i).track(:,3))
end

%% nice 3D plot
figure
for i=1:length(inbees)
hold on
plot3(inbees(i).track(:,1),inbees(i).track(:,2),inbees(i).track(:,3)*-1,'b')
end
for i=1:length(outbees)
hold on
plot3(outbees(i).track(:,1),outbees(i).track(:,2),outbees(i).track(:,3)*-1,'g')
end
for i=1:length(hornet3D)
hold on
plot3(hornet3D(i).track(:,1),hornet3D(i).track(:,2),hornet3D(i).track(:,3)*-1,'r')
end
disp(toc)


%% END OF THE LINE
%{

for i=1:length(hornet3D)
hold on
plot3(hornet3D(i).track(:,1),hornet3D(i).track(:,2),hornet3D(i).track(:,3))
end

vertcat(DATA(i).topU(abs(bothCams(1)-DATA(i).starttop+1):abs(bothCams(2)-DATA(i).starttop+1),:))
DATA(j).sideU(abs(bothCams(1)-DATA(j).startside+1):abs(bothCams(2)-DATA(j).startside+1),:)


DATA(i).topU(abs(bothCams(1)-DATA(i).starttop+1):abs(bothCams(2)-DATA(i).starttop+1))
DATA(j).sideU(abs(bothCams(1)-DATA(i).startside+1):abs(bothCams(2)-DATA(i).startside+1))


m=1;
for i=1:1270
    if DATA(i).directiontop == 0 
        if DATA(i).durtop > 20
        DataHtop(n,:)=[DATA(i).starttop DATA(i).durtop DATA(i).endtop i]
        n=n+1;
        end
    end
    if DATA(i).directionside == 0 
        if DATA(i).durside > 20
        DataHside(m,:)=[DATA(i).startside DATA(i).durside DATA(i).endside i]
        m=m+1;
        end
    end
    
end


n=1;
for i=1:length(DataHside)
for j=1:length(DataHtop)
    if (DataHside(i,1)>DataHtop(j,1) && DataHside(i,1)<DataHtop(j,3)) || (DataHtop(j,1)>DataHside(i,1) && DataHtop(j,3)<DataHside(i,1))
       bothCams= [max([DataHside(i,1) DataHtop(j,1)])  min([DataHside(i,3) DataHtop(j,3)])];
    end
end
end

        
             [q qq qqq]=find(vertcat(DATA.starttop)>bothCams(1,1),1)
   
        
        if DataHside(i,3)<DataHtop(j,3) %later and ended before
    triH(n).side=DATA(DataHside(i,4)).sideU;
    triH(n).top=DATA(DataHtop(i,4)).topU();    
        else   % later but longer
            
    end
    
end
end






nowTop=DATA(i).starttop;
durTop=DATA(i).durtop;

[iSide distance]=dsearchn(vertcat(DATA(1:180).startside),nowTop);
nowSide=DATA(iSide).startside;



bees3D(i,1:lengthtop,:) = triangulate(topcoo,sidecoo,stereoParams2018); 

u=Xtop(i,:);
u(isnan(u))=[]
u=isfinite(Xtop(i,:));



% get the hornet out of bees
hornet=nan(6,size(bees,3));
for i=1:videoFrames
    [M,I]=max(bees(3,:,i));
    if M>150
    hornet(1:3,i)=squeeze(bees(1:3,I,i));
    bees(1:3,I,i)=nan;
    end
    [M,I]=max(bees(6,:,i));
    if M > 200
    hornet(4:6,i)=squeeze(bees(4:6,I,i));
    bees(4:6,I,i)=nan;
    end
end

amountoftime=10000;
figure
top=(beesFull(1:3,:,1:amountoftime));
side=(bees(4:6,:,1:amountoftime));
bees3D=nan(281,size(bees,2),3);
for i=1:amountoftime
 %   isfinite(Q_loc_estimateX(10,1:20))
    lengthtop=sum(isfinite((top(1,:,i))));
    lengthside=sum(isfinite((side(1,:,i))));
    topcoo=squeeze(top(1:2,1:lengthtop,i));
    sidecoo=squeeze(side(1:2,1:lengthside,i));
    
    if lengthtop == lengthside && lengthtop ~=0
        topcoo(1,:)=(topcoo(1,:)+rectBehaviour(1)-0)*4.167;
        topcoo(2,:)=(topcoo(2,:)+rectBehaviour(2)-0)*3.125;        
        sidecoo(1,:)=(sidecoo(1,:)+rectBehaviour2(1)-0)*4.167;
        sidecoo(2,:)=(sidecoo(2,:)+rectBehaviour2(2)-0)*3.125;
        sidecoo=undistortPoints(sidecoo',stereoParams2018.CameraParameters2);
        topcoo=undistortPoints(topcoo',stereoParams2018.CameraParameters1);
        bees3D(i,1:lengthtop,:) = triangulate(topcoo,sidecoo,stereoParams2018);   
    end
end
bees3D(bees3D==0)=nan;
plot3(bees3D(:,:,2)',bees3D(:,:,3)',bees3D(:,:,1)','*')
hold on
showExtrinsics(stereoParams2018);


view(90,0)

figure
plot(squeeze(bees(1,:,1:amountoftime)),squeeze(bees(2,:,1:amountoftime)),'*')


return % only compile until here!

% ab hier neu schauen was man gebrauchen kann

bees2=cat(3,nan(6,38,1),bees2);  %the first one is nans
bees2Sort=nan(6,40,100);
k=1; %k erste runde
cam1ready=0;
for i=2:100
if sum(isfinite(bees2(1,:,i-1)))==0 && sum(isfinite(bees2(1,:,i)))==1 
     cam1ready=1;
end
if sum(isfinite(bees2(4,:,i-1)))==0 && sum(isfinite(bees2(4,:,i)))==1 
     i
end
    
end


plot3(bees3D(:,2,1),bees3D(:,2,2),bees3D(:,2,3))

    
    [maxSize indexMax] = max(bees2(3,:,i));
bees2Sort(1:3,k,i)=bees2(1:3,indexMax);
bees2(1:3,indexMax)=nan;

[maxSize indexMax] = max(bees2(3,:,i));
    if  maxSize> 200
        
        hornet2(1:3,i)=bees2(1:3,indexMax,i);
    end


        
[indexofClosest, distanz] = knnsearch(bees(1,:,i+1)',sortedBees(1,j,i));


%%ende baustelle

beesOhneH=bees;

disp(toc)
disp('done hornet finding')

% make hornet 2x 2D points undistord (lens error cam) and then fuse them to
% 3D due to checkerboard data / stereo cams
point3d=nan(videoFrames,4);
for i=1:videoFrames

hornet1=([hornet(1,i)+rectBehaviour(1), hornet(2,i)+rectBehaviour(2)]);
hornet2=([hornet(4,i)+rectBehaviour(3), hornet(5,i)+rectBehaviour(4)]);
if isfinite(hornet1(1)) &&  isfinite(hornet2(1))
undistortedPoints1 = undistortPoints(hornet1,stereoParamsWEI.CameraParameters1);
undistortedPoints2 = undistortPoints(hornet2,stereoParamsWEI.CameraParameters2);
else
    undistortedPoints1=[0 0];
    undistortedPoints2=[0 0];
end
point3d(i,1:3) = triangulate(undistortedPoints1, undistortedPoints2, stereoParamsWEI);
point3d(i,4) = norm(point3d)/1000;

end

disp(toc)
disp('3D-ing the hornet')

sortedbeesAll=nan(6,200,videoFrames);

bees=beesOhneH;
sortedBees=nan(size(bees));
sortedBees(:,:,1)=bees(:,:,1);
figure
hold on
for o=1:10   %too small, better while bees is finite...
   sortedbeesAll(:,(1+((o-1)*size(bees,2))):(size(bees,2)*o),:)=sortedBees;
    sortedBees=nan(size(bees));
    sortedBees(:,:,1)=bees(:,:,1);  
    for i=2:videoFrames-1
        if sum(isnan(bees(1,:,i-1))) == size(bees,2) && sum(isnan(bees(1,:,i))) < size(bees,2)
            sortedBees(1:3,:,i)=bees(1:3,:,i);
            %bees(1:3,:,i)=nan;
        end
    end
    bees(isfinite(sortedBees))=nan;
    for i=1:videoFrames-1
        for j=1:size(bees,2)
            if isfinite(sortedBees(1,j,i))
                [indexofClosest, distanz] = knnsearch(bees(1,:,i+1)',sortedBees(1,j,i));
                %   for k=1:20
                if distanz(1) < 20
                    % if abs(bees(1,k,i+1)- sortedBees(1,j,i)) < 20
                    sortedBees(1:3,j,i+1)=bees(1:3,indexofClosest(1),i+1);
                    bees(1:3,indexofClosest(1),i+1)=nan;
                end
            end
        end
    end
    
    plot(squeeze(sortedbeesAll(1,(1+((o-1)*size(bees,2))):(size(bees,2)*o),:))',squeeze(sortedbeesAll(2,(1+((o-1)*size(bees,2))):(size(bees,2)*o),:))','*')
   
end


bees=beesOhneH;
sortedBees=nan(size(bees));
sortedBees(:,:,1)=bees(:,:,1);
figure
hold on
for o=1:10
     sortedbeesAll(:,(1+((o-1)*size(bees,2))):(size(bees,2)*o),:)=sortedBees;
    sortedBees=nan(size(bees));
    sortedBees(:,:,1)=bees(:,:,1);
    for i=2:videoFrames-1
        if sum(isnan(bees(4,:,i-1))) == size(bees,2) && sum(isnan(bees(4,:,i))) < size(bees,2)
            sortedBees(4:6,:,i)=bees(4:6,:,i);
            %bees(1:3,:,i)=nan;
        end
    end
    bees(isfinite(sortedBees))=nan;
    
    for i=1:videoFrames-1
        for j=1:size(bees,2)
            if isfinite(sortedBees(4,j,i))
                [indexofClosest, distanz] = knnsearch(bees(4,:,i+1)',sortedBees(4,j,i));
                %   for k=1:20
                if distanz(1) < 20
                    % if abs(bees(1,k,i+1)- sortedBees(1,j,i)) < 20
                    sortedBees(4:6,j,i+1)=bees(4:6,indexofClosest(1),i+1);
                    bees(4:6,indexofClosest(1),i+1)=nan;
                end
            end
        end
    end
    
    plot(squeeze(sortedbeesAll(4,(1+((o-1)*size(bees,2))):(size(bees,2)*o),:))',squeeze(sortedbeesAll(5,(1+((o-1)*size(bees,2))):(size(bees,2)*o),:))','*')

end


k=1;
for i=1:200
   if sum(sum(isfinite(sortedbeesAll(:,i,:)))) >0
    sortedbeesAll2(:,k,:)=sortedbeesAll(:,i,:);
    k=k+1;
    
   end
  
end

disp(toc)
disp('all data realighed and code finished')




plot(squeeze(sortedbeesAll2(4,:,:))','*')


justBees=nan(6,size(sortedbeesAll,2),videoFrames);

for o=1:size(sortedbeesAll,2)
for i=1:videoFrames-1

if isfinite(sortedbeesAll(1,o,i)) && isfinite(sortedbeesAll(1,o,i+1))


end
end
end



   if max(bees(1,:,i)) > 500 && max(bees(4,:,i)) > 570
    % here startes a new track coming from the outside
    i
   end
    max(bees(4,:,i)) 
    max(bees(1,:,i))


%I1 = undistortImage(readFrame(v),stereoParamsWEI.CameraParameters1);
%I2 = undistortImage(readFrame(v2),stereoParamsWEI.CameraParameters2);


cam1bees=bees(1:3,:,:);
cam2bees=bees(4:6,:,:);

cam1bees(isnan(cam2bees))=nan;
cam2bees(isnan(cam1bees))=nan;



%   if sum(isfinite(bees(1,:,i))) == 1 && sum(isfinite(bees(1,:,i-1))) == 0


sortMatrix=nan(1000,20,2);
for i=1:100
    
    beeRow1=bees(1,:,i);
    beeRow1(isnan(beeRow1))=[];
    beeRow2=bees(2,:,i);
    beeRow2(isnan(beeRow2))=[];
    beeRow3=bees(3,:,i);
    beeRow3(isnan(beeRow3))=[];
    beeRow4=bees(4,:,i);
    beeRow4(isnan(beeRow4))=[];
    beeRow5=bees(5,:,i);
    beeRow5(isnan(beeRow5))=[];
    beeRow6=bees(6,:,i);
    beeRow6(isnan(beeRow6))=[];
    smallerArray(1) = length(beeRow1);
    smallerArray(2) = length(beeRow4);
    minRow=min(smallerArray); % the smaller of both so only if both cams have a coordiante
    beeRow1_min=beeRow1(1:minRow);
    beeRow2_min=beeRow2(1:minRow);
    beeRow3_min=beeRow3(1:minRow);
    beeRow4_min=beeRow4(1:minRow);
    beeRow5_min=beeRow5(1:minRow);
    beeRow6_min=beeRow6(1:minRow);
    
    if i>1
        BeeField2=BeeField;
    end
    
    BeeField=[beeRow1_min', beeRow2_min'];%, beeRow3_min];  , beeRow4_min', beeRow5_min', beeRow6_min']';
    
    if i>1
        [Idx,D] = knnsearch(BeeField,BeeField2);
        sortMatrix(i,1:length(Idx),1)=Idx;
        sortMatrix(i,1:length(Idx),2)=D;    
    end
    
end

return


beesTest=bees;

beesTest(isnan(bees))=[];

a=knnsearch(squeeze(bees(1:2,:,1)),squeeze(bees(1:2,:,2)))

load hospital;
X = [hospital.Age hospital.Weight];
Y = [20 162; 30 169; 40 168; 50 170; 60 171]; 

Idx = knnsearch(X,Y);

X(Idx,:)


% find individual continuous tracks of bees

entrypoints=squeeze(max(bees(2,:,:)));
entrypoints2=squeeze(max(bees(5,:,:)));
j=1;
for i=2:videoFrames
    if (isnan(entrypoints(i-1)) || isnan(entrypoints2(i-1))) && (isfinite(entrypoints(i)) && isfinite(entrypoints2(i)))
        startpoints(j)=i;
        j=j+1;
    end
end

h=1;
for i=1:length(startpoints)
 [M,I]=max(bees(1,:,startpoints(i)));
 beeID()=squeeze(bees(1:3,:,startpoints(i)));


end

load('stereoParamWEI.mat')      % camera config from stereocal
 stereoParams = stereoParamsWEI;
showExtrinsics(stereoParams);


plot([96 987],[7779 37139])
hold on
plot([0 8266],[1000 8266])


% level(i) = graythresh(BGvideo(rect(2):rect(2)+rect(4),rect(1):rect(1)+rect(3),i)-BG(rect(2):rect(2)+rect(4),rect(1):rect(1)+rect(3)))

hold on
for i=1:12
 plot(squeeze(bees(1,i,:)),squeeze(bees(2,i,:)),'*')
 plot(squeeze(hornets(1,i,:)),squeeze(hornets(2,i,:)),'o')
%plot(squeeze(hornets(3,i,:)),'o')
%hist(squeeze(bees(3,:,:)),1000)

end

bothanimals=[bees,hornets];

%}


