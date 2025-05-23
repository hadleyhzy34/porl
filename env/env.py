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
    def __init__(agent, action_size, state_size, rank_update_interval = 150, namespace='tb3'):
        super().__init__()
        agent.action_space = spaces.Discrete(action_size)
        # agent.observation_space = spaces.Sequence(state_size)
        agent.observation_space = spaces.Discrete(state_size)
        agent.observation_space = spaces.Box(low=-np.inf,high=np.inf,shape=(362,),dtype=np.float32)
        agent.goal_x = 0
        agent.goal_y = 0
        agent.namespace = namespace
        agent.status = 'initialized'
        agent.map_x = -10.
        agent.map_y = -10.
        agent.rank = 0
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

        agent.min_range = 0.13
        agent.cur_step = 0
        agent.episode_step = 500
        agent.cur_episode = 0
        agent.rank_update_interval = rank_update_interval

    def getGoalDistace(agent):
        goal_distance = round(math.hypot(agent.goal_x - agent.position.x, agent.goal_y - agent.position.y), 2)

        return goal_distance

    def getOdometry(agent, odom):
        # print(f'odometry is called')
        agent.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)

        goal_angle = math.atan2(agent.goal_y - agent.position.y, agent.goal_x - agent.position.x)

        heading = goal_angle - yaw
        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        agent.heading = round(heading, 2)

    def getState(agent, scan):
        # import ipdb;ipdb.set_trace()
        scan_range = []
        heading = agent.heading
        done = False

        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf'):
                scan_range.append(3.5)
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])

        # print(f'min range is: {min(scan_range)}')
        if agent.min_range > min(scan_range) > 0:
            # print(f'hit!')
            done = True # done because of collision

        current_distance = round(math.hypot(agent.goal_x - agent.position.x, agent.goal_y - agent.position.y),2)
        if current_distance < 0.2:
            agent.get_goalbox = True
            done = True # done because of goal reached

        # print(f'current agent position: {agent.position.x}, {agent.position.y}')
        return scan_range + [heading, current_distance], done

    def setReward(agent, state, done, action):
        # pdb.set_trace()
        yaw_reward = []
        current_distance = state[-1]
        heading = state[-2]

        for i in range(5):
            angle = -pi / 4 + heading + (pi / 8 * i) + pi / 2
            tr = 1 - 4 * math.fabs(0.5 - math.modf(0.25 + 0.5 * angle % (2 * math.pi) / math.pi)[0])
            yaw_reward.append(tr)

        distance_rate = 2 ** (current_distance / agent.goal_distance)
        reward = ((round(yaw_reward[action] * 5, 2)) * distance_rate)

        if reward < -100 or reward > 100:
            # pdb.set_trace()
            print(f'distance rate is: {distance_rate}, '
                  f'yaw_reward is: {yaw_reward[action]}')
            reward = reward
        if done:
            if agent.get_goalbox: # done because of goal reached
                # rospy.loginfo("Goal!")
                reward = 200
                agent.pub_cmd_vel.publish(Twist())
                agent.get_goalbox = False
                agent.status = 'goal'
            else:  # done because of collision
                # rospy.loginfo("Collision!!")
                reward = -200
                agent.pub_cmd_vel.publish(Twist())
                agent.status = 'hit'

        return reward

    def step(agent, action):
        max_angular_vel = 1.5
        ang_vel = ((agent.action_size - 1)/2 - action) * max_angular_vel * 0.5

        vel_cmd = Twist()
        vel_cmd.linear.x = 0.15
        vel_cmd.angular.z = ang_vel
        agent.pub_cmd_vel.publish(vel_cmd)
        # exec_time = time.time()

        # waiting more or less equal to 0.2 s until scan data received, stable behavior
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message(agent.namespace+'/scan', LaserScan, timeout=5)
            except:
                pass

        # print(f'waiting scan data duration is: {time.time() - exec_time}')
        state, done = agent.getState(data)
        reward = agent.setReward(state, done, action)

        agent.cur_step += 1
        truncated = False
        if agent.cur_step >= agent.episode_step:
            truncated = True
            agent.status = 'timelimited'

        # print(f'cur_step:{agent.cur_step}, '
        #     f'reward:{reward:.5f}, '
        #       f'done:{done}, '
        #       f'truncated:{truncated}')
        return np.asarray(state), reward, done, truncated, {"status": agent.status}

    def collect_step(agent, action):
        vel_cmd = Twist()
        vel_cmd.linear.x = action[0]
        vel_cmd.angular.z = action[1]
        agent.pub_cmd_vel.publish(vel_cmd)
        # exec_time = time.time()

        # waiting more or less equal to 0.2 s until scan data received, stable behavior
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message(agent.namespace+'/scan', LaserScan, timeout=5)
            except:
                pass

        # print(f'waiting scan data duration is: {time.time() - exec_time}')
        state, done = agent.getState(data)
        reward = agent.setReward(state, done, action)

        agent.cur_step += 1
        truncated = False
        if agent.cur_step >= agent.episode_step:
            truncated = True
            agent.status = 'timelimited'

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
        agent.goal_distance = agent.getGoalDistace()
        state, done = agent.getState(data)

        agent.cur_step = 0
        agent.cur_episode += 1

        if agent.cur_episode % agent.rank_update_interval == 0:
            agent.rank += 1

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
        while dist < 0.25 or dist > 3.5:
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
