from os import truncate
import pdb
import rospy
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState
import numpy as np
import math
from math import pi
import time
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import gymnasium as gym
from gymnasium import spaces
# import gym
# from gym import spaces
import numpy as np
from stable_baselines3.common.env_checker import check_env

class Env(gym.Env):
    def __init__(agent, namespace, state_size, action_size, rank = 0):
        super().__init__()
        # agent.action_space = spaces.Box(low=-1.,high=1.,shape=(2,),dtype=np.float32)
        agent.action_space = spaces.Discrete(action_size)
        agent.observation_space = spaces.Box(low=-np.inf,high=np.inf,shape=(362,),dtype=np.float32)
        agent.goal_x = 0
        agent.goal_y = 0
        agent.namespace = namespace
        agent.status = 'initialized'
        agent.map_x = -10.
        agent.map_y = -10.
        agent.rank = rank
        agent.heading = 0
        agent.action_size = action_size
        agent.initGoal = True
        agent.get_goalbox = False  # reach goal or not
        agent.position = Pose()
        agent.pub_cmd_vel = rospy.Publisher(agent.namespace+'/cmd_vel', Twist, queue_size=5)
        agent.sub_odom = rospy.Subscriber(agent.namespace+'/odom', Odometry, agent.getOdometry)
        agent.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        agent.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        agent.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        # agent.respawn_goal = Respawn()
        #
        # physics of tb3
        agent.max_lin_vel = 0.15
        agent.max_ang_vel = 1.5

        agent.min_range = 0.13
        agent.cur_step = 0
        agent.episode_step = 500
        agent.cur_episode = 0

    def getRelativeGoal(agent):
        """
        Description: return relative goal distance and angle based on robot frame
        """
        # relative goal position based on robot base frame
        goal_pos = np.array([agent.goal_x - agent.position.x,
                             agent.goal_y - agent.position.y])

        goal_distance = np.linalg.norm(goal_pos)

        # relative goal position based on robot base frame
        rot = np.array([[np.cos(agent.heading),np.sin(agent.heading)],
                        [-np.sin(agent.heading),np.cos(agent.heading)]])  #(2,2)
        goal_pose = np.matmul(rot, goal_pos[:,None])[:,0]  #(2,)

        goal_angle = np.abs(np.arctan2(goal_pose[1],goal_pose[0]))

        # assert np.abs(goal_distance - np.linalg.norm(goal_pose)) < 1e-5, "goal pose distance should be kept same after rotation"

        return goal_distance, goal_angle, goal_pose

    def lidarPreprocess(agent, scan):
        # remove inf/nan data from lidar scan
        scan_range = np.array(scan.ranges)
        scan_range[np.isnan(scan_range)] = 10.
        scan_range[np.isinf(scan_range)] = 10.

        return scan_range

    def getOdometry(agent, odom):
        agent.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, agent.heading = euler_from_quaternion(orientation_list)

    def getSarsa(agent,scan):
        """
        Description:
        args:
            scan: list[360]
        return:
            state: np.array, (362,)
        """
        terminated = False
        reward = 0.

        goal_distance, goal_angle, goal_pose = agent.getRelativeGoal()
        state = agent.getState(scan, goal_pose)

        # goal distance reward
        if goal_distance < agent.goal_distance:
            reward += agent.goal_distance - goal_distance
        else:
            reward += 2 * (agent.goal_distance - goal_distance)
        agent.goal_distance = goal_distance

        # goal heading reward
        if goal_angle < agent.goal_angle:
            reward += agent.goal_angle - goal_angle
        else:
            reward += 2 * (agent.goal_angle - goal_angle)
        agent.goal_angle = goal_angle

        # check collision
        if agent.min_range > np.min(state[:360]) > 0:
            terminated = True # done because of Collision
            reward = -500.
            agent.pub_cmd_vel.publish(Twist())
            agent.status = 'hit'

        # check if reached goal
        if goal_distance < 0.2:
            terminated = True
            reward = 500.
            agent.pub_cmd_vel.publish(Twist())
            agent.status = 'goal'

        return state, reward, terminated

    def getState(agent, scan, goal_pose):
        """
        Description:
        args:
            scan: list[360]
            goal_pose: np.array(2,)
        return:
            state: np.array, (362,)
        """
        # pdb.set_trace()
        scan_range = agent.lidarPreprocess(scan)

        return np.concatenate((scan_range, goal_pose), axis=0)

    def step(agent, action):
        '''
        Description: env step() function
        args:
            action: (2,)
        return:
        '''
        vel_cmd = Twist()
        vel_cmd.linear.x = action[0]
        vel_cmd.angular.z = action[1]
        agent.pub_cmd_vel.publish(vel_cmd)

        # waiting more or less equal to 0.2 s until scan data received, stable behavior
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message(agent.namespace+'/scan', LaserScan, timeout=5)
            except:
                pass

        state, reward, done = agent.getSarsa(data)

        agent.cur_step += 1
        truncated = False
        if agent.cur_step >= agent.episode_step:
            truncated = True

        # print(f'cur_step:{agent.cur_step}, '
        #       f'reward:{reward:.5f}, '
        #       f'done:{done}, '
        #       f'truncated:{truncated}')

        return np.asarray(state), reward, done, truncated, {"status": agent.status}

    def collect_step(agent, action):
        '''
        Description: env step() function
        args:
            action: (2,)
        return:
        '''
        vel_cmd = Twist()
        vel_cmd.linear.x = action[0]
        vel_cmd.angular.z = action[1]
        agent.pub_cmd_vel.publish(vel_cmd)

        # waiting more or less equal to 0.2 s until scan data received, stable behavior
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message(agent.namespace+'/scan', LaserScan, timeout=5)
            except:
                pass

        state, reward, done = agent.getSarsa(data)

        agent.cur_step += 1
        truncated = False
        if agent.cur_step >= agent.episode_step:
            truncated = True

        return np.asarray(state), reward, done, truncated, {"status": agent.status}

    def render(agent):
        pass

    def close(agent):
        pass

    def reset(agent, seed=None, options=None):
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            agent.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")

        # pdb.set_trace()
        agent.reset_model() #reinitialize model starting position

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message(agent.namespace+'/scan', LaserScan, timeout=5)
            except:
                pass

        if agent.initGoal:
            # agent.goal_x, agent.goal_y = agent.respawn_goal.getPosition()
            # _,_,agent.goal_x, agent.goal_y = agent.random_pts_map()
            agent.initGoal = False

        agent.status = 'running'
        agent.goal_distance, agent.goal_angle, goal_pose = agent.getRelativeGoal()
        state = agent.getState(data, goal_pose)

        agent.cur_step = 0
        agent.cur_episode += 1

        return np.asarray(state), {"status": agent.status}

    def reset_model(agent):
        state_msg = ModelState()
        state_msg.model_name = agent.namespace
        # update initial position and goal positions
        state_msg.pose.position.x, state_msg.pose.position.y, agent.goal_x, agent.goal_y = agent.random_pts_map()
        agent.init_x = state_msg.pose.position.x
        agent.init_y = state_msg.pose.position.y
        # state_msg.pose.position.z = 0.3
        state_msg.pose.orientation.x = 0
        state_msg.pose.orientation.y = 0
        state_msg.pose.orientation.z = 0
        state_msg.pose.orientation.w = 1.

        # modify target cricket ball position
        target = ModelState()
        target.model_name = 'target_red_0'
        target.pose.position.x = agent.goal_x
        target.pose.position.y = agent.goal_y
        target.pose.position.z = 0.
        # target.pose.position.x = 0.
        # target.pose.position.y = 0.

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            set_state( state_msg )
            set_state( target )

        except (rospy.ServiceException) as e:
            print("gazebo/set_model_state Service call failed")

    def random_pts_map(agent):
        """
        Description: random initialize starting position and goal position, make distance > 0.1
        return:
            x1,y1: initial position
            x2,y2: goal position
        """
        # pdb.set_trace()
        # print(f'goal position is going to be reset')
        x1,x2,y1,y2 = 0,0,0,0
        while (x1 - x2) ** 2 < 0.01:
            x1 = np.random.randint(5) + np.random.uniform(0.16, 1-0.16) # random initialize x inside single map
            x2 = np.random.randint(5) + np.random.uniform(0.16, 1-0.16) # random initialize goal position

        while (y1 - y2) ** 2 < 0.01:
            y1 = np.random.randint(5) + np.random.uniform(0.16, 1-0.16) # random initialize y inside single map
            y2 = np.random.randint(5) + np.random.uniform(0.16, 1-0.16) # random initialize goal position

        x1 = agent.map_x + (agent.rank % 4) * 5 + x1
        y1 = agent.map_y + (3 - agent.rank // 4) * 5 + y1

        x2 = agent.map_x + (agent.rank % 4) * 5 + x2
        y2 = agent.map_y + (3 - agent.rank // 4) * 5 + y2

        # set waypoint within scan range
        # pdb.set_trace()
        dist = np.linalg.norm([y2 - y1, x2 - x1]) 
        while dist < 0.3 or dist > 3.5:
            x2 = np.random.randint(5) + np.random.uniform(0.16, 1-0.16) # random initialize goal position
            y2 = np.random.randint(5) + np.random.uniform(0.16, 1-0.16) # random initialize goal position

            x2 = agent.map_x + (agent.rank % 4) * 5 + x2
            y2 = agent.map_y + (3 - agent.rank // 4) * 5 + y2
            dist = np.linalg.norm([y2 - y1, x2 - x1])

        # agent.rank = (agent.rank + 1) % 8
        # print(f'goal position is respawned')

        return x1,y1,x2,y2

if __name__ == "__main__":
    env = Env(5,362)
    check_env(env)
