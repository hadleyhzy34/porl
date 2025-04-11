import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pdb
from dataloader.a_star import AStarPlanner
from matplotlib import pyplot as plt


class CustomDataset(Dataset):
    def __init__(agent, device=torch.device('cpu')):
        agent.device = device
        agent.data_length = 365 + 1 + 365 + 1 + 2

        cur_data = torch.zeros((100,361))
        count = 0
        file_count = 0
        for f in os.listdir('checkpoint/'):
            data = np.loadtxt(f, delimiter=',',dtype=float)
            data = data.reshape(-1,agent.data_length)
            for j in range(data.shape[0]):
                save_data = preprocessing(data[j])
                if save_data is not None:
                    cur_data[count] = save_data
                    count += 1
                    if count % 100 == 0:
                        tgt_file = 'checkpoint/a_star/dataset_' + str(file_count) + '.pt'
                        torch.save(cur_data.view(-1,370), tgt_file)
                        file_count += 1
                        cur_data = torch.zeros((100,361))

        agent.length = count

    def __getitem__(agent, index):
        pdb.set_trace()
        file_index = index // 100
        row_index = index % 100
        # data = np.loadtxt(f"checkpoint/dataset_{file_index}.csv", delimiter=',',dtype=float)
        data = torch.load(f'checkpoint/a_star/dataset_{file_index}.pt')[row_index]
        data = data.reshape(-1,agent.data_length)[row_index]

        # preprocessing(data)
        # data = np.concatenate([
        #     data['state'],
        #     [data['reward']],
        #     data['next_state'],
        #     [data['done']],
        #     data['action']
        #     ])
        data = torch.from_numpy(data.astype(np.float32)).to(agent.device)

        return data

    def __len__(agent):
        return agent.length


# def preprocessing(data_dir, tgt_dir, file_list, length_list):
def preprocessing(data):
    """
    Args:
        data: (s,r,d,s,a)
    """
    pdb.set_trace()

    # generate expert policy
    grid_size = .1
    robot_radius = .13
    show_animation = True

    if data[:360].min() < robot_radius:
        return

    sx = 0.
    sy = 0.
    heading = data[362]
    goal_pose = data[363:365] - data[360:362]  #(2,)

    # relative goal position based on robot base frame
    rot = np.array([[np.cos(heading),np.sin(heading)],
                    [-np.sin(heading),np.cos(heading)]])  #(2,2)
    goal_pose = np.matmul(rot, goal_pose[:,None])[:,0]  #(2,)
    gx = goal_pose[0]
    gy = goal_pose[1]

    ox = []
    oy = []
    for i in range(360):
        if data[i] < 3.5 and data[i] > 0.15:
            ox.append(np.cos(i*np.pi/180)*data[i])
            oy.append(np.sin(i*np.pi/180)*data[i])
        if show_animation:  # pragma: no cover
            plt.plot(ox, oy, ".k")
            plt.plot(sx, sy, "og")
            plt.plot(gx, gy, "xb")
            plt.grid(True)
            plt.axis("equal")

    # pdb.set_trace()
    a_star = AStarPlanner(ox, oy, grid_size, robot_radius)
    rx, ry = a_star.planning(sx, sy, gx, gy)
    # print(f'len of r: {len(rx)}')

    goal_r = 15.
    value = goal_r * np.power(0.99,len(rx))
    pdb.set_trace()
    data = np.concatenate([data[:360],[value]])

    if show_animation:  # pragma: no cover
        plt.plot(rx, ry, "-r")
        plt.pause(0.001)
        plt.show()

    return data

    # # pdb.set_trace()
    # tgt_file = tgt_dir + '/' + filename[:-4] + '.pt'
    # torch.save(cur_data.view(-1,370), tgt_file)
    #
    # with open(file_list,'a') as f:
    #     np.savetxt(f, np.array([tgt_file]), fmt='%s',delimiter=',')
    #
    # with open(length_list,'a') as f:
    #     np.savetxt(f, np.array([total_length]), fmt='%d',delimiter=',')
    #
    # total_length += cur_data.shape[0]

def general_process():
    cur_data = torch.zeros((100,361))
    count = 0
    file_count = 0

    data_length = 365 + 1 + 365 + 1 + 2
    for f in os.listdir('checkpoint/'):
        data = np.loadtxt(f, delimiter=',',dtype=float)
        data = data.reshape(-1,data_length)
        for j in range(data.shape[0]):
            save_data = preprocessing(data[j])
            if save_data is not None:
                cur_data[count] = save_data
                count += 1
                if count % 100 == 0:
                    tgt_file = 'checkpoint/a_star/dataset_' + str(file_count) + '.pt'
                    torch.save(cur_data.view(-1,370), tgt_file)
                    file_count += 1
                    cur_data = torch.zeros((100,361))

if __name__ == "__main__":
    general_process()
