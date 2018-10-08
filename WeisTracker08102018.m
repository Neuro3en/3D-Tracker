clear
close all
tic
profile on
tracklength=30;   % how long must a track be at least in frames (4ms each)
tracklength2=10;   % how long must a track be at least in frames (4ms each)
simultanBees=50;  % how many bees at the same time do you expect? give it 5x due to noise
expectedTracks=2000;
BeeSizeThreashold=50;
BeeSizeThreashold2=50;
LEDflash=2000;
videoFrames=20000;     % MOST IMORTANT hjow many frames to analyze 
                        %(video duration in seconds * 250)-2000
videoframerounds=1;
leftEdge = 1500;  % pixel threashold on the left, for in and outbound definition
rightEdge = 3000;
hornetDurationTOP=400; % how long a hornet must be visible to be detected
hornetDurationSIDE=200;
%  v.CurrentTime = 0.6;   % go to particular timepoint  at 30FPS

% stART VIDEO AND GET LIGHT TRIGGER
v = VideoReader('Top88.MP4');
imshow(readFrame(v));
rectSync = round(getrect);
v2 = VideoReader('Side26.MP4');
imshow(readFrame(v2));
rectSync2 = round(getrect);

% behaviour ROI
imshow(readFrame(v));
rectBehaviour = getrect;
rectBehaviour = round(rectBehaviour);
imshow(readFrame(v2));
rectBehaviour2 = getrect;
rectBehaviour2 = round(rectBehaviour2);
close all

load('stereoParamWei2018.mat')      % camera config from stereocal

% find trigger 1
pic2sync3=readFrame(v);
pic2sync=pic2sync3(:,:,2);
syncPic=sum(sum(pic2sync(rectSync(2):rectSync(2)+rectSync(4),rectSync(1)...
    :rectSync(1)+rectSync(3))));
for i=1:3000
    pic2sync3_1=readFrame(v);
    pic2sync_1=pic2sync3_1(:,:,2);
    syncPic_1=sum(sum(pic2sync_1(rectSync(2):rectSync(2)+rectSync(4),rectSync(1)...
        :rectSync(1)+rectSync(3))));
    if syncPic_1-syncPic > LEDflash
        % imshow(pic2sync(rectSync(2):rectSync(2)+rectSync(4),rectSync(1)...
        % :rectSync(1)+rectSync(3)));
        break
    end
    syncPic=syncPic_1;
end

% find trigger 2
pic2sync3=readFrame(v2);
pic2sync=pic2sync3(:,:,2);
syncPic=sum(sum(pic2sync(rectSync2(2):rectSync2(2)+rectSync2(4),rectSync2(1)...
    :rectSync2(1)+rectSync2(3))));
for i=1:3000
    pic2sync3_1=readFrame(v2);
    pic2sync_1=pic2sync3_1(:,:,2);
    syncPic_1=sum(sum(pic2sync_1(rectSync2(2):rectSync2(2)+rectSync2(4),rectSync2(1)...
        :rectSync2(1)+rectSync2(3))));
    if syncPic_1-syncPic > LEDflash
        % imshow(pic2sync(rectSync2(2):rectSync2(2)+rectSync2(4),rectSync2(1)...
        % :rectSync2(1)+rectSync2(3)));
        break
    end
    syncPic=syncPic_1;
end

% jump over LEDflash
for i=1:200
    readFrame(v);
    readFrame(v2);
end

%   Background for substraction
BGvideo=uint8(zeros(720,1280,10));
j=1;

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

%{
 imshow(BG);
 figure
 imshow(BG-BGvideo(:,:,1));
  if threshold to noisy or no bees
for i=1:99
 level(i) = graythresh(BG(rect(2):rect(2)+rect(4),rect(1):rect(1)+rect(3))...
-BGvideo(rect(2):rect(2)+rect(4),rect(1):rect(1)+rect(3),i))
end
 will be commentet out later
 for i=1:99
 imshow(imbinarize(BG(rect(2):rect(2)+rect(4),rect(1):rect(1)+rect(3))...
-BGvideo(rect(2):rect(2)+rect(4),rect(1):rect(1)+rect(3),i)));
  drawnow
 end
%}


disp('go')

for rounds=1:videoframerounds
    
    bees=zeros(6,simultanBees,videoFrames);
    for i=1:videoFrames
        currentFrame = readFrame(v);
        if mod(i,10) == 0
            BGvideo(:,:,rem(i,10)+1) = currentFrame(:,:,2);
            BG=median(BGvideo(:,:,:),3);
        end
        BWcurrent = imbinarize(BG(rectBehaviour(2):rectBehaviour(2)...
            +rectBehaviour(4),rectBehaviour(1):rectBehaviour(1)+...
            rectBehaviour(3))-currentFrame(rectBehaviour(2): ...
            rectBehaviour(2)+rectBehaviour(4),rectBehaviour(1): ...
            rectBehaviour(1)+rectBehaviour(3),2));
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
        BWcurrent = imbinarize(BG2(rectBehaviour2(2):rectBehaviour2(2)+ ...
            rectBehaviour2(4),rectBehaviour2(1):rectBehaviour2(1)+ ...
            rectBehaviour2(3))-currentFrame(rectBehaviour2(2): ...
            rectBehaviour2(2)+rectBehaviour2(4),rectBehaviour2(1): ...
            rectBehaviour2(1)+rectBehaviour2(3),2));
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
    
    bees(bees==0)=nan;
    beesfull=bees;
    %% if there are no tracks delete, time gab!!
    for i=1:videoFrames-1
        if (sum(sum(isfinite(bees(1,:,i))))==0 || sum(sum(isfinite(bees(4,:,i))))==0)
            bees(:,:,i)=[];
        end
        if i==size(bees,3) || i==size(bees,3)-1 || i==size(bees,3)-1
            break
        end
    end
    
  %  toc
    disp('ended tracking')
    
    %% here the monteCarloCode for top
    
    
framesinBee=size(bees,3);
%framesinBee=3000;
Y = cell(1,framesinBee);         
    X = cell(1,framesinBee);         
    n=1;
    for i=1:framesinBee
        aX=squeeze(bees(1,:,i));
        aY=squeeze(bees(2,:,i));
        aX(isnan(aX))=[];
        aY(isnan(aY))=[];       
        if ~isempty(aX)
        X{n}=aX';
        Y{n}=aY'; 
        n=n+1;
        end
    end
    
    % define main variables for KALMAN FILTER! :P
    dt = 1;  %our sampling rate
    S_frame = 5; %find(cellfun(@length, X)>0,1); %starting frame
    
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
    Q_estimate = nan(4,expectedTracks);
    Q_estimate(:,1:size(Q,2)) = Q;  %estimate of initial location estimation of where the flies are(what we are updating)
    Q_loc_estimateY = nan(size(bees,3),expectedTracks); %  position estimate
    Q_loc_estimateX= nan(size(bees,3),expectedTracks); %  position estimate
    P_estimate = P;  %covariance estimator
    strk_trks = zeros(1,expectedTracks);  %counter of how many strikes a track has gotten
    nD = size(X{S_frame},1); %initize number of detections
    nF =  find(isnan(Q_estimate(1,:))==1,1)-1 ; %initize number of track estimates
    
    %for each frame
    for t = S_frame:n-1 %framesinBee-1;         
        
        Q_loc_meas = [X{t} Y{t}];
        
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
    
    rej = [];
    for F = 1:nF
        if asgn(F) > 0
            rej(F) =  est_dist(F,asgn(F)) < 50 ;
        else
            rej(F) = 0;
        end
    end
    asgn = asgn.*rej;
    
            % asgn(isempty(asgn))=0;
            % ok, now we check for tough situations and if it's tough, just go with estimate and ignore the data
            %make asgn = 0 for that tracking element
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
                
                %strk_trks = zeros(1,(no_trk_list(end)));
                if ~isempty(no_trk_list)
                %     
                    strk_trks(no_trk_list) = strk_trks(no_trk_list) + 1;
                else
                  %  strk_trks=nan;
                end
                
                %if a track has a strike greater than 6, delete the tracking. i.e.
                %make it nan first vid = 3
                bad_trks = find(strk_trks > 6);
                Q_estimate(:,bad_trks) = NaN;

    end
    
    disp('done with 1. monteCarlo')
    
    %% here the monteCarloCode for side
    
 
    
framesinBee=size(bees,3);
%framesinBee=3000;
Y = cell(1,framesinBee);         
    X = cell(1,framesinBee);         
    n=1;
    for i=1:framesinBee
        aX=squeeze(bees(4,:,i));
        aY=squeeze(bees(5,:,i));
        aX(isnan(aX))=[];
        aY(isnan(aY))=[];       
        if ~isempty(aX)
        X{n}=aX';
        Y{n}=aY'; 
        n=n+1;
        end
    end
    
    % define main variables for KALMAN FILTER! :P
    dt = 1;  %our sampling rate
    S_frame = find(cellfun(@length, X)>0,1); %starting frame
    
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
    Q_estimate = nan(4,expectedTracks);
    Q_estimate(:,1:size(Q,2)) = Q;  %estimate of initial location estimation of where the flies are(what we are updating)
    Q_loc_estimateY2 = nan(size(bees,3),expectedTracks); %  position estimate
    Q_loc_estimateX1= nan(size(bees,3),expectedTracks); %  position estimate
    P_estimate = P;  %covariance estimator
    strk_trks = zeros(1,expectedTracks);  %counter of how many strikes a track has gotten
    nD = size(X{S_frame},1); %initize number of detections
    nF =  find(isnan(Q_estimate(1,:))==1,1)-1 ; %initize number of track estimates
    
    %for each frame
    for t = S_frame:n-1 %framesinBee-1;         
        
        Q_loc_meas = [X{t} Y{t}];
        
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
    
    rej = [];
    for F = 1:nF
        if asgn(F) > 0
            rej(F) =  est_dist(F,asgn(F)) < 50 ;
        else
            rej(F) = 0;
        end
    end
    asgn = asgn.*rej;
    
            % asgn(isempty(asgn))=0;
            % ok, now we check for tough situations and if it's tough, just go with estimate and ignore the data
            %make asgn = 0 for that tracking element
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
                
                %strk_trks = zeros(1,(no_trk_list(end)));
                if ~isempty(no_trk_list)
                %     
                    strk_trks(no_trk_list) = strk_trks(no_trk_list) + 1;
                else
                  %  strk_trks=nan;
                end
                
                %if a track has a strike greater than 6, delete the tracking. i.e.
                %make it nan first vid = 3
                bad_trks = find(strk_trks > 6);
                Q_estimate(:,bad_trks) = NaN;

    end

    disp('done with 2. monteCarlo')
    
    %% matchCamera realWorldValues
    
    Xtop=(Q_loc_estimateX1+rectBehaviour(1))*4.167;
    Ytop=(Q_loc_estimateY2+rectBehaviour(2))*3.125;
    Xside=(Q_loc_estimateX+rectBehaviour(1))*4.167;
    Yside=(Q_loc_estimateY+rectBehaviour(2))*3.125;
    
    clear DATA
    
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
    end
    
    
    %% folowing only if below comented
    for i=1:size(Xtop,2)
        [DATA(i).topU]=[DATA(i).Xtop,DATA(i).Ytop];
        [DATA(i).sideU]=[DATA(i).Xside,DATA(i).Yside];
    end
    
    %{
for i=1:videoFrames
    if ~isempty(DATA(i).Xtop) && ~isempty(DATA(i).Xside)
[DATA(i).topU]=undistortPoints([DATA(i).Xtop,DATA(i).Ytop],stereoParams2018.CameraParameters1);
[DATA(i).sideU]=undistortPoints([DATA(i).Xside,DATA(i).Yside],stereoParams2018.CameraParameters2);
    end
end
DATA=rmfield(DATA,{'Xtop','Ytop','Xside','Yside'});
    %}
    
    
    
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
        if DATA(i).durtop < tracklength
            DATA(i).durtop=nan;
            DATA(i).topU=nan;
            DATA(i).starttop=nan;
            DATA(i).endtop=nan;
        end
        if DATA(i).durside < tracklength
            DATA(i).durside=nan;
            DATA(i).sideU=nan;
            DATA(i).startside=nan;
            DATA(i).endside=nan;
        end
    end
    
    %% inbound vs outbound marking
    for i=1:length(DATA)
        if ~isempty(DATA(i).topU)
            if DATA(i).topU(1,1)<leftEdge && DATA(i).topU(end,1)> rightEdge  % out
                DATA(i).directiontop=1;
            end
        end
        if ~isempty(DATA(i).topU)
            if DATA(i).topU(1,1)>leftEdge && DATA(i).topU(end,1)<rightEdge  % in
                DATA(i).directiontop=-1;
            end
        end
        if ~isempty(DATA(i).sideU)
            if DATA(i).sideU(1,1)<leftEdge && DATA(i).sideU(end,1)>rightEdge  % out
                DATA(i).directionside=1;
            end
        end
        if ~isempty(DATA(i).sideU)
            if DATA(i).sideU(1,1)>leftEdge && DATA(i).sideU(end,1)<rightEdge  % in
                DATA(i).directionside=-1;
            end
        end
    end
    
    
    %% hornet
    %  figure
    %  hold on
    for i=1:length(DATA)
        if isfinite(DATA(i).durtop)
            if DATA(i).durtop > hornetDurationTOP && isempty(DATA(i).directiontop)  % 500
                %         plot(DATA(i).topU(:,1),DATA(i).topU(:,2),'r')
                DATA(i).directiontop=0;
            end
        end
        if  isfinite(DATA(i).durside)
            if DATA(i).durside > hornetDurationSIDE && isempty(DATA(i).directionside)  % 200
                %        plot(DATA(i).sideU(:,1),DATA(i).sideU(:,2),'b')
                DATA(i).directionside=0;
            end
        end
    end
    
    
    %% hornet traj with DATA
    clear hornet3D
    n=1;
    for i=1:length(DATA)
        if DATA(i).directiontop == 0
            if DATA(i).durtop > tracklength2
                for j=1:length(DATA)
                    if DATA(j).directionside == 0
                        if DATA(j).durside > tracklength2
                            if (DATA(i).starttop > DATA(j).startside && ...
                                    DATA(i).starttop < DATA(j).endside) || ...
                                    (DATA(j).startside > DATA(i).starttop && ...
                                    DATA(j).startside < DATA(i).endtop)
                                bothCams= [max([DATA(i).starttop DATA(j).startside])  ...
                                    min([DATA(i).endtop  DATA(j).endside])];
                                hornet3D(n).track=triangulate ...
                                    (DATA(i).topU(abs(bothCams(1)-DATA(i).starttop+1): ...
                                    abs(bothCams(2)-DATA(i).starttop+1),:), ...
                                    DATA(j).sideU(abs(bothCams(1)-DATA(j).startside+1): ...
                                    abs(bothCams(2)-DATA(j).startside+1),:),stereoParams2018);
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
    %{
    figure
    for i=1:length(hornet3D)
        hold on
        plot3(hornet3D(i).track(:,1),hornet3D(i).track(:,2),hornet3D(i).track(:,3))
    end
    %}
    
    %% trajec bees  OUTBOUND
    clear outbees
    n=1;
    for i=1:length(DATA)
        if DATA(i).directiontop == 1
            if DATA(i).durtop > tracklength2
                for j=1:length(DATA)
                    if DATA(j).directionside == 1
                        if DATA(j).durside > tracklength2
                            if (DATA(i).starttop > DATA(j).startside && ...
                                    DATA(i).starttop < DATA(j).endside) || ...
                                    (DATA(j).startside > DATA(i).starttop && ...
                                    DATA(j).startside < DATA(i).endtop)
                                bothCams= [max([DATA(i).starttop DATA(j).startside])  ...
                                    min([DATA(i).endtop  DATA(j).endside])];
                                outbees(n).track=triangulate ...
                                    (DATA(i).topU(abs(bothCams(1)-DATA(i).starttop+1):...
                                    abs(bothCams(2)-DATA(i).starttop+1),:),...
                                    DATA(j).sideU(abs(bothCams(1)-DATA(j).startside+1):...
                                    abs(bothCams(2)-DATA(j).startside+1),:),stereoParams2018);
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
    
    %{
    figure
    for i=1:length(outbees)
        hold on
        plot3(outbees(i).track(:,1),outbees(i).track(:,2),outbees(i).track(:,3))
        pause
    end
    %}
    
    %% trajec bees  INBOUND
    clear inbees
    n=1;
    for i=1:length(DATA)
        if DATA(i).directiontop == -1
            if DATA(i).durtop > tracklength2
                for j=1:length(DATA)
                    if DATA(j).directionside == -1
                        if DATA(j).durside > tracklength2
                            if (DATA(i).starttop > DATA(j).startside && ...
                                    DATA(i).starttop < DATA(j).endside) || ...
                                    (DATA(j).startside > DATA(i).starttop && ...
                                    DATA(j).startside < DATA(i).endtop)
                                bothCams= [max([DATA(i).starttop DATA(j).startside])  ...
                                    min([DATA(i).endtop  DATA(j).endside])];
                                inbees(n).track=triangulate ...
                                    (DATA(i).topU(abs(bothCams(1)-DATA(i).starttop+1):...
                                    abs(bothCams(2)-DATA(i).starttop+1),:),...
                                    DATA(j).sideU(abs(bothCams(1)-DATA(j).startside+1):...
                                    abs(bothCams(2)-DATA(j).startside+1),:),stereoParams2018);
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
    %{
    figure
    for i=1:length(inbees)
        hold on
        plot3(inbees(i).track(:,1),inbees(i).track(:,2),inbees(i).track(:,3))
    end
    %}
    
    %% trajec bees  OTHER
    clear otherbees
    n=1;
    for i=1:length(DATA)
        if isempty(DATA(i).directiontop)
            if DATA(i).durtop > tracklength2
                for j=1:length(DATA)
                    if isempty(DATA(i).directionside)
                        if DATA(j).durside > tracklength2
                            if (DATA(i).starttop > DATA(j).startside && ...
                                    DATA(i).starttop < DATA(j).endside) || ...
                                    (DATA(j).startside > DATA(i).starttop && ...
                                    DATA(j).startside < DATA(i).endtop)
                                bothCams= [max([DATA(i).starttop DATA(j).startside])  ...
                                    min([DATA(i).endtop  DATA(j).endside])];
                                otherbees(n).track=triangulate ...
                                    (DATA(i).topU(abs(bothCams(1)-DATA(i).starttop+1): ...
                                    abs(bothCams(2)-DATA(i).starttop+1),:),...
                                    DATA(j).sideU(abs(bothCams(1)-DATA(j).startside+1):...
                                    abs(bothCams(2)-DATA(j).startside+1),:),stereoParams2018);
                                otherbees(n).start=bothCams(1);
                                otherbees(n).end=bothCams(2);
                                n=n+1;
                            end
                        end
                    end
                end
            end
        end
    end
        
    
    if ~exist('otherbees')
        outbees=0;
    else
    disp('other bees')
    disp(length(otherbees))
    end
    
    if ~exist('hornet3D')
        hornet3D=0;
    else
    disp('hornets coordinates')
    disp(length(vertcat(hornet3D.track)))
    end
    
    if ~exist('inbees')
        inbees=0;
    else
    disp('inbound bees')
    disp(length(inbees))
    end
    
    if ~exist('outbees')
        outbees=0;
    else
    disp('outbound bees')
    disp(length(outbees))
    end



    
    
    %{
    figure
    for i=1:length(otherbees)
        hold on
        plot3(otherbees(i).track(:,1),otherbees(i).track(:,2),otherbees(i).track(:,3))
    end
    %}
    

    
    
    %% nice 3D plot
    %{
    figure
    for i=1:length(inbees)
        hold on
        plot3(inbees(i).track(:,1),inbees(i).track(:,2),inbees(i).track(:,3),'b')
    end
    for i=1:length(outbees)
        hold on
        plot3(outbees(i).track(:,1),outbees(i).track(:,2),outbees(i).track(:,3),'g')
    end
    for i=1:length(hornet3D)
        hold on
        plot3(hornet3D(i).track(:,1),hornet3D(i).track(:,2),hornet3D(i).track(:,3),'r')
    end
    for i=1:length(otherbees)
        hold on
        plot3(otherbees(i).track(:,1),otherbees(i).track(:,2),otherbees(i).track(:,3),'c')
    end
    %}

    
    
    if rounds==1
        save('Tracks1.mat','hornet3D','inbees','outbees','DATA','otherbees','bees')
    elseif rounds==2
        save('Tracks2.mat','hornet3D','inbees','outbees','DATA','otherbees','bees')
    elseif rounds==3
        save('Tracks3.mat','hornet3D','inbees','outbees','DATA','otherbees','bees')
    elseif rounds==4
        save('Tracks4.mat','hornet3D','inbees','outbees','DATA','otherbees','bees')
    elseif rounds==5
        save('Tracks5.mat','hornet3D','inbees','outbees','DATA','otherbees','bees')
    elseif rounds==6
        save('Tracks6.mat','hornet3D','inbees','outbees','DATA','otherbees','bees')
    elseif rounds==7
        save('Tracks7.mat','hornet3D','inbees','outbees','DATA','otherbees','bees')
    elseif rounds==8
        save('Tracks8.mat','hornet3D','inbees','outbees','DATA','otherbees','bees')
    elseif rounds==9
        save('Tracks9.mat','hornet3D','inbees','outbees','DATA','otherbees','bees')
    elseif rounds==10
        save('Tracks10.mat','hornet3D','inbees','outbees','DATA','otherbees','bees')
    end
    toc
    disp(rounds)
    
end
profile viewer











