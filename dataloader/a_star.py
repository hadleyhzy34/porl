import math
import pdb
import matplotlib.pyplot as plt

show_animation = False


class AStarPlanner:

    def __init__(agent, ox, oy, resolution, rr):
        """
        Initialize grid map for a star planning
        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        resolution: grid resolution [m]
        rr: robot radius[m]
        """

        agent.resolution = resolution
        agent.rr = rr
        agent.min_x, agent.min_y = 0, 0
        agent.max_x, agent.max_y = 0, 0
        agent.obstacle_map = None
        agent.x_width, agent.y_width = 0, 0
        agent.motion = agent.get_motion_model()
        agent.calc_obstacle_map(ox, oy)

    class Node:
        def __init__(agent, x, y, cost, parent_index):
            agent.x = x  # index of grid
            agent.y = y  # index of grid
            agent.cost = cost
            agent.parent_index = parent_index

        def __str__(agent):
            return str(agent.x) + "," + str(agent.y) + "," + str(
                agent.cost) + "," + str(agent.parent_index)

    def planning(agent, sx, sy, gx, gy):
        """
        A star path search
        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]
        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """

        start_node = agent.Node(agent.calc_xy_index(sx, agent.min_x),
                               agent.calc_xy_index(sy, agent.min_y), 0.0, -1)
        goal_node = agent.Node(agent.calc_xy_index(gx, agent.min_x),
                              agent.calc_xy_index(gy, agent.min_y), 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[agent.calc_grid_index(start_node)] = start_node

        while 1:
            if len(open_set) == 0:
                # pdb.set_trace()
                print("Open set is empty..")
                break

            c_id = min(
                open_set,
                key=lambda o: open_set[o].cost + agent.calc_heuristic(goal_node,
                                                                     open_set[
                                                                         o]))
            current = open_set[c_id]

            # show graph
            if show_animation:  # pragma: no cover
                plt.plot(agent.calc_grid_position(current.x, agent.min_x),
                         agent.calc_grid_position(current.y, agent.min_y), "xc")
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event',
                                             lambda event: [exit(
                                                 0) if event.key == 'escape' else None])
                if len(closed_set.keys()) % 10 == 0:
                    plt.pause(0.001)

            if current.x == goal_node.x and current.y == goal_node.y:
                print("Find goal")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand_grid search grid based on motion model
            for i, _ in enumerate(agent.motion):
                node = agent.Node(current.x + agent.motion[i][0],
                                 current.y + agent.motion[i][1],
                                 current.cost + agent.motion[i][2], c_id)
                n_id = agent.calc_grid_index(node)

                # If the node is not safe, do nothing
                if not agent.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # discovered a new node
                else:
                    if open_set[n_id].cost > node.cost:
                        # This path is the best until now. record it
                        open_set[n_id] = node

        # pdb.set_trace()
        rx, ry = agent.calc_final_path(goal_node, closed_set)

        return rx, ry

    def calc_final_path(agent, goal_node, closed_set):
        # generate final course
        rx, ry = [agent.calc_grid_position(goal_node.x, agent.min_x)], [
            agent.calc_grid_position(goal_node.y, agent.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(agent.calc_grid_position(n.x, agent.min_x))
            ry.append(agent.calc_grid_position(n.y, agent.min_y))
            parent_index = n.parent_index

        return rx, ry

    @staticmethod
    def calc_heuristic(n1, n2):
        w = 1.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def calc_grid_position(agent, index, min_position):
        """
        calc grid position
        :param index:
        :param min_position:
        :return:
        """
        pos = index * agent.resolution + min_position
        return pos

    def calc_xy_index(agent, position, min_pos):
        return round((position - min_pos) / agent.resolution)

    def calc_grid_index(agent, node):
        return (node.y - agent.min_y) * agent.x_width + (node.x - agent.min_x)

    def verify_node(agent, node):
        px = agent.calc_grid_position(node.x, agent.min_x)
        py = agent.calc_grid_position(node.y, agent.min_y)

        if px < agent.min_x:
            return False
        elif py < agent.min_y:
            return False
        elif px >= agent.max_x:
            return False
        elif py >= agent.max_y:
            return False

        # collision check
        if agent.obstacle_map[node.x][node.y]:
            return False

        return True

    def calc_obstacle_map(agent, ox, oy):

        # agent.min_x = round(min(ox))
        # agent.min_y = round(min(oy))
        # agent.max_x = round(max(ox))
        # agent.max_y = round(max(oy))
        agent.min_x = -10.
        agent.min_y = -5.
        agent.max_x = 10.
        agent.max_y = 5.
        # print("min_x:", agent.min_x)
        # print("min_y:", agent.min_y)
        # print("max_x:", agent.max_x)
        # print("max_y:", agent.max_y)

        agent.x_width = round((agent.max_x - agent.min_x) / agent.resolution)
        agent.y_width = round((agent.max_y - agent.min_y) / agent.resolution)
        # print("x_width:", agent.x_width)
        # print("y_width:", agent.y_width)

        # obstacle map generation
        agent.obstacle_map = [[False for _ in range(agent.y_width)]
                             for _ in range(agent.x_width)]
        for ix in range(agent.x_width):
            x = agent.calc_grid_position(ix, agent.min_x)
            for iy in range(agent.y_width):
                y = agent.calc_grid_position(iy, agent.min_y)
                for iox, ioy in zip(ox, oy):
                    d = math.hypot(iox - x, ioy - y)
                    if d <= agent.rr:
                        agent.obstacle_map[ix][iy] = True
                        break

    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]

        return motion


def main():
    print(__file__ + " start!!")

    # start and goal position
    sx = 10.0  # [m]
    sy = 10.0  # [m]
    gx = 50.0  # [m]
    gy = 50.0  # [m]
    grid_size = 2.0  # [m]
    robot_radius = 1.0  # [m]

    # set obstacle positions
    ox, oy = [], []
    for i in range(-10, 60):
        ox.append(i)
        oy.append(-10.0)
    for i in range(-10, 60):
        ox.append(60.0)
        oy.append(i)
    for i in range(-10, 61):
        ox.append(i)
        oy.append(60.0)
    for i in range(-10, 61):
        ox.append(-10.0)
        oy.append(i)
    for i in range(-10, 40):
        ox.append(20.0)
        oy.append(i)
    for i in range(0, 40):
        ox.append(40.0)
        oy.append(60.0 - i)

    if show_animation:  # pragma: no cover
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "og")
        plt.plot(gx, gy, "xb")
        plt.grid(True)
        plt.axis("equal")

    a_star = AStarPlanner(ox, oy, grid_size, robot_radius)
    rx, ry = a_star.planning(sx, sy, gx, gy)
    # pdb.set_trace()

    if show_animation:  # pragma: no cover
        plt.plot(rx, ry, "-r")
        plt.pause(0.001)
        plt.show()
