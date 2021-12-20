%**********************************************************************%
%*************************--Q-LEARNING--*******************************%
%*************************----ROBOT----********************************%
%************************---OMID FAIZI---******************************%
%*********************---F.SOMID@YMAIL.COM---**************************%
%**********************************************************************%

%% initilize:

clc
clear
close all

%% initial variables:

rect = [100, 100]; % size of field rectangle: W * h
iter = 5000; % number of maximum iteration for learning
loop = 100; % maximum iteration for each run in learning
Ts = 0.1; % time step (second)
goal_v = 0*[0.03; 0.1]; % speed of moving goal
n_obs = 7; % number of obstacles
obs_v = 0*[0.1, -0.7, 0.01, -0.1, -0.05, 0.1, -0.3;
         -0.3, 0.05, 0.05, -0.05, -0.05, -0.05, -0.05]; % velocity of obstacles
start_pos = [pi/4; 1; 1];
robot = zeros(3, loop, iter); % robot specification (v,phi,x,y)
robot_v = 10; % initial velocity of robot (constant speed)
% initial phi, x, y of robot in each learning loop:
robot(1:3, 1, :) = repmat(start_pos, 1, iter); 
sensor_range = 5; % sensor range should be integer

%% figure:

figure('name', 'Reinforcement Learning', 'numberTitle', 'off')
axis([-15 100 -15 100])
box on
hold on
xlabel('X')
ylabel('Y')

%% main loop:

Q = 100+zeros(145, sensor_range*4+1, 37, 25); % Q-learning matrix
% first state is distnace from goal, second is obstacle position, third is
% goal angle (pi/18 section) & action is angle of moving (pi/12 section)
epsilon = 0.3; % this is for epsilon-greedy algorithm
action = 0;
reward = zeros(loop, iter);
alpha = 0.7; % learning factor
gamma = 0.8; % discount factor

for i=1:iter
    
    goal_pos = [10; 20]; % initial position of goal
    obs_pos = [20,30,10,40,60,10,70;
                  20,20,10,50,65,60,70]; % initial obstacle position
    d_goal = zeros(loop, 1); % distance from goal
    d_obs = zeros(n_obs, loop); % distance from obstacle
    near_obs = zeros(loop, 1); % nearest obstacle
    state = zeros(loop,3); % empty state matrix
    action = zeros(loop,1); % empty action matrix
    
    for j=1:loop
        d_goal(j) = sqrt( ( robot(2,j,i) - goal_pos(1) ).^2 +...
            ( robot(3,j,i) - goal_pos(2) ).^2 ); % distance from goal
        
        phi_goal = atan2(( robot(3,j,i) - goal_pos(2) ),...
            ( robot(2,j,i) - goal_pos(1) )); % goal angle
        
        if phi_goal < 0 
            phi_goal = phi_goal + pi;
        end
        
        state(j,1) = max( floor(d_goal(j)), 1); % state of goal
        
        d_obs(:,j) = sqrt( ( robot(2,j,i) - obs_pos(1,:) ).^2 +...
            ( robot(3,j,i) - obs_pos(2,:) ).^2 );
        if any(d_obs(:,j) < sensor_range)
            % create angle of obstacles and choose nearest for each sensor
            % choose the nearest obstacle as state
            [~,near_obs(j)] = min(d_obs(:,j));
            phi_obs = atan2(obs_pos(2,near_obs(j))-robot(3,j,i),...
                obs_pos(1,near_obs(j))-robot(2,j,i));
            if phi_obs < 0 
                phi_obs = phi_obs + pi;
            end
            
            phi_obs = phi_obs - robot(1,j,i);
            if phi_obs < 0
                phi_obs = phi_obs + 2*pi;
            end
            
            if phi_obs>=0 && phi_obs<pi/2
                % front sensor: 1
                state(j,2) = (floor(d_obs(near_obs(j),j))+1) + 0;
                
            elseif phi_obs>=pi/2 && phi_obs<pi
                % right sensor: 2
                state(j,2) = (floor(d_obs(near_obs(j),j))+1) + 5;
                
            elseif phi_obs>=pi && phi_obs<3*pi/2
                % back sensor: 3
                state(j,2) = (floor(d_obs(near_obs(j),j))+1) + 10;
                
            elseif phi_obs>=3*pi/2 && phi_obs<=2*pi
                % left sensor: 4
                state(j,2) = (floor(d_obs(near_obs(j),j))+1) + 15;
            end
        else
            % new state that means nothing around robot
            state(j,2) = sensor_range*4+1;
        end
        
        
        % state of goal phi:
        state(j,3) = floor(phi_goal/(pi/18)) + 1;
        
        % take action based on states (epsilon-greedy)
        rnd = rand;
        if rnd < epsilon
            acti = randi(25);
        else
            max_value = max(Q(state(j,1),state(j,2),state(j,3),:));
            acti = find(Q(state(j,1),state(j,2),state(j,3),:)==max_value);
        end
        
        if length(acti)>1
            action(j) = acti( randi( length(acti) ) );
        else
            action(j) = acti;
        end
        
        robot(1,j+1,i) = (action(j)-1)*pi/12;
        
        
        % reward value
        if j == 1
            reward(j,i) = 5*abs(robot(1,j+1,i) - phi_goal - pi)/(pi);
            
        elseif near_obs(j)==0
            reward(j,i) = (d_goal(j-1) - d_goal(j));
            
        else
            reward(j,i) = (d_goal(j-1) - d_goal(j)) +...
                (d_obs(near_obs(j),j-1) - d_obs(near_obs(j),j));
        end
        
        
        % update states (j+1)
        
        robot(2,j+1,i) = robot_v*cos(robot(1,j,i))*Ts + robot(2,j,i);
        robot(3,j+1,i) = robot_v*sin(robot(1,j,i))*Ts + robot(3,j,i);
        
        obs_pos = obs_pos + obs_v;
        if any(obs_pos(1,:) > rect(1) | obs_pos(1,:) < 0)
            obs_v(1, obs_pos(1,:) > rect(1) | obs_pos(1,:) < 0) = ...
                -obs_v(1, obs_pos(1,:) > rect(1) | obs_pos(1,:) < 0);
        end
        
        if any(obs_pos(2,:) > rect(2) | obs_pos(2,:) < 0)
            obs_v(2, obs_pos(2,:) > rect(2) | obs_pos(2,:) < 0) = ...
                -obs_v(2, obs_pos(2,:) > rect(2) | obs_pos(2,:) < 0);
        end
        
        goal_pos = goal_pos + goal_v;
        if goal_pos(1) > rect(1) || goal_pos(1) < 0
            goal_v(1) = -goal_v(1);
        end
        
        if goal_pos(2) > rect(2) || goal_pos(2) < 0
            goal_v(2) = -goal_v(2);
        end
        
        % update Q matrix
        d_goal(j+1) = sqrt( ( robot(2,j,i) - goal_pos(1) ).^2 +...
            ( robot(3,j,i) - goal_pos(2) ).^2 ); % distance from goal
        
        n_state(1) = max( floor(d_goal(j+1)), 1); % state of goal
        
        d_obs(:,j+1) = sqrt( ( robot(2,j+1,i) - obs_pos(1,:) ).^2 +...
            ( robot(3,j+1,i) - obs_pos(2,:) ).^2 );
        
        if any(d_obs(:,j+1) < sensor_range)
            % create angle of obstacles and choose nearest for each sensor
            % choose the nearest obstacle as state
            [~,near_obs(j+1)] = min(d_obs(:,j+1));
            phi_obs = atan2(obs_pos(2,near_obs(j+1))-robot(3,j+1,i),...
                obs_pos(1,near_obs(j+1))-robot(2,j+1,i));
            if phi_obs < 0 
                phi_obs = phi_obs + pi;
            end
            
            phi_obs = phi_obs - robot(1,j,i);
            if phi_obs < 0
                phi_obs = phi_obs + 2*pi;
            end
            
            
            if phi_obs>=0 && phi_obs<pi/2
                % front sensor: 1
                n_state(2) = (floor(d_obs(near_obs(j+1),j+1))+1) * 1;
            elseif phi_obs>=pi/2 && phi_obs<pi
                % right sensor: 2
                n_state(2) = (floor(d_obs(near_obs(j+1),j+1))+1) * 2;
            elseif phi_obs>=pi && phi_obs<3*pi/2
                % back sensor: 3
                n_state(2) = (floor(d_obs(near_obs(j+1),j+1))+1) * 3;
            elseif phi_obs>=3*pi/2 && phi_obs<2*pi
                % left sensor: 4
                n_state(2) = (floor(d_obs(near_obs(j+1),j+1))+1) * 4;
            end
        else
            % new state that means nothing around robot
            n_state(2) = sensor_range*4+1;
        end
        % state of goal phi:
        n_state(3) = floor(phi_goal/(pi/18)) + 1;
        
        neighbour = max( Q( n_state(1), n_state(2), n_state(3), : ) );
        
        Q(state(j,1),state(j,2),state(j,3),action(j)) = ...
            Q( state(j,1), state(j,2), state(j,3), action(j) ) + ...
            alpha*( reward(j,i) + gamma * neighbour - ...
            Q( state(j,1), state(j,2), state(j,3), action(j) ) );
        
        if j==loop && d_goal(j) > 5 && i>5
            Q(state(:,1), state(:,2), state(j,3), action(:)) = ...
                Q(state(:,1), state(:,2), state(j,3), action(:))...
                - min(Q(state(:,1), state(:,2), state(j,3), action(:)) , [] , 'all');
            
        elseif j<loop && d_goal(j) < 1
            Q(state(1:j,1), state(1:j,2), state(1:j,3), action(1:j)) = ...
                Q(state(1:j,1), state(1:j,2), state(1:j,3), action(1:j))...
                - max(Q(state(1:j,1), state(1:j,2), state(1:j,3), action(1:j)) , [] , 'all'); 
            break
        end
        
        disp('iteration:   reward:   goal distance:')
        disp([num2str(loop*(i-1)+j),'         ',...
            num2str(reward(j,i)),'       ',num2str(d_goal(j))])
        
        % figure:
        cla
        plot(goal_pos(1),goal_pos(2),'ro','MarkerSize',10,'MarkerFaceColor','r')
        plot(obs_pos(1,:),obs_pos(2,:),'b*','MarkerSize',10,'MarkerFaceColor','c')
        plot(robot(2,j,i),robot(3,j,i),'ks','MarkerSize',10,'MarkerFaceColor','c')
        title({['iteration: ',num2str(loop*(i-1)+j)],...
            ['reward: ',num2str(reward(j,i))],...
            ['goal distance: ',num2str(d_goal(j))]})
        % legend('Goal','Obstacles','Robot')
        drawnow
    end
    
end

