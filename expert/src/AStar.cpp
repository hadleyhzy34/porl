#include "AStar.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <math.h>
#include <unordered_map>
#include <utility>

AStar::AStar(float res, float rad, struct World map)
    : resolution(res), radius(rad), worldMap(map) {
  motionVector = {
      {1, 0, 1},         {0, 1, 1},          {-1, 0, 1},
      {0, -1, 1},        {-1, -1, sqrtf(2)}, {-1, 1, sqrtf(2)},
      {1, -1, sqrtf(2)}, {1, 1, sqrtf(2)},
  };

  this->worldMap.width.first =
      (this->worldMap.X.second - this->worldMap.X.first) / this->resolution;
  this->worldMap.width.second =
      (this->worldMap.Y.second - this->worldMap.Y.first) / this->resolution;

  this->obsMap = std::vector<std::vector<bool>>(
      this->worldMap.width.first,
      std::vector<bool>(this->worldMap.width.second, false));

  std::cout << "AStar is created:[radius] " << this->radius << " [resolution] "
            << this->resolution << " [minX] " << this->worldMap.X.first
            << " [maxX] " << this->worldMap.X.second << " [minY] "
            << this->worldMap.Y.first << " [maxY] " << this->worldMap.Y.second
            << " [xWidth] " << this->worldMap.width.first << " [yWidth] "
            << this->worldMap.width.second << std::endl;
}

AStar::~AStar() {}

float AStar::calcGridPosition(int index, float minPosition) {
  return index * this->resolution + minPosition;
}

int AStar::calcGridIndex(struct Node node) {
  return node.yId * this->worldMap.width.first + node.xId;
  /* return (node.y - this->worldMap.minY) * this->worldMap.xWidth + */
  /*        (node.x - this->worldMap.minX); */
}

int AStar::calcXYIndex(float position, float minPosition) {
  return std::round((position - minPosition) / this->resolution);
}

float AStar::calcHeuristic(struct Node n1, struct Node n2) {
  return std::sqrt(std::pow((n1.x - n2.x), 2) + std::pow((n1.y - n2.y), 2));
}

void AStar::calcObsMap(float ox, float oy) {
  for (auto i = 0; i < this->worldMap.width.first; i++) {
    float x = calcGridPosition(i, this->worldMap.X.first);
    for (auto j = 0; j < this->worldMap.width.second; j++) {
      float y = calcGridPosition(j, this->worldMap.Y.first);
      float d = std::pow(x - ox, 2) + std::pow(y - oy, 2);
      if (d < std::pow(this->radius, 2)) {
        this->obsMap[i][j] = true;
      }
    }
  }
}

bool AStar::checkNode(struct Node node) {
  if (node.x < this->worldMap.X.first) {
    return false;
  } else if (node.y < this->worldMap.Y.first) {
    return false;
  } else if (node.x >= this->worldMap.X.second) {
    return false;
  } else if (node.y >= this->worldMap.Y.second) {
    return false;
  }

  if (this->obsMap[node.xId][node.yId]) {
    return false;
  }

  return true;
}

void AStar::plan(float ox, float oy, float gx, float gy) {
  struct Node startNode = {this->calcXYIndex(ox, this->worldMap.X.first),
                           this->calcXYIndex(oy, this->worldMap.Y.first),
                           ox,
                           oy,
                           0.,
                           -1};

  struct Node endNode = {this->calcXYIndex(gx, this->worldMap.X.first),
                         this->calcXYIndex(gy, this->worldMap.Y.first),
                         gx,
                         gy,
                         0.,
                         -1};

  std::unordered_map<int, struct Node> openSet;
  std::unordered_map<int, struct Node> closedSet;

  openSet[this->calcGridIndex(startNode)] = startNode;
  int count = 0;

  while (true) {
    if (openSet.empty()) {
      std::cout << "open set is empty";
      break;
    }

    auto iter = std::min_element(
        openSet.begin(), openSet.end(),
        [this, &endNode](const auto &l, const auto &r) {
          return l.second.cost + this->calcHeuristic(l.second, endNode) <
                 r.second.cost + this->calcHeuristic(r.second, endNode);
        });
    struct Node curNode = iter->second;
    int curId = iter->first;

    if (curNode.xId == endNode.xId && curNode.yId == endNode.yId) {
      std::cout << "goal found";
      endNode.parentIdx = curNode.parentIdx;
      endNode.cost = curNode.cost;
      break;
    }

    count++;
    std::cout << curNode.x << "||" << curNode.y << "||" << curNode.xId << "||"
              << curNode.yId << "||"
              << curNode.cost + this->calcHeuristic(curNode, endNode) << "||"
              << openSet.size() << "||" << closedSet.size() << "||" << count
              << std::endl;

    openSet.erase(curId);
    closedSet[curId] = curNode;

    // grid search
    for (auto motion : this->motionVector) {
      struct Node newNode = {
          this->calcXYIndex(curNode.x + motion[0] * this->resolution,
                            this->worldMap.X.first),
          this->calcXYIndex(curNode.y + motion[1] * this->resolution,
                            this->worldMap.Y.first),
          curNode.x + motion[0] * this->resolution,
          curNode.y + motion[1] * this->resolution,
          curNode.cost + motion[2] * this->resolution,
          curId};
      int nodeId = this->calcGridIndex(newNode);

      // check if this node is safe
      if (!this->checkNode(newNode)) {
        continue;
      }
      // check if this node is found in closed_set
      if (closedSet.find(nodeId) != closedSet.end()) {
        continue;
      }
      // check if this node is found in openSet
      if (openSet.find(nodeId) == openSet.end()) {
        openSet[nodeId] = newNode;
      } else {
        if (openSet[nodeId].cost > newNode.cost) {
          openSet[nodeId] = newNode;
        }
      }
    }
  }

  this->calcPath(endNode, closedSet);
}

void AStar::calcPath(struct Node node,
                     std::unordered_map<int, struct Node> closedSet) {
  this->path.push_back(std::make_pair(node.x, node.y));
  int parentIdx = node.parentIdx;

  while (parentIdx != -1) {
    struct Node n = closedSet[parentIdx];
    this->path.push_back(std::make_pair(n.x, n.y));
    parentIdx = n.parentIdx;
  }
}

void AStar::printPath() {
  if (this->path.empty()) {
    return;
  }
  for (auto p : this->path) {
    std::cout << p.first << " " << p.second << std::endl;
  }
}

/* std::ostream &operator<<(std::ostream &os, */
/*                          const std::unordered_map<int, struct Node> &map) {
 */
/*   for (auto x : map) { */
/*     os << x.first << " " << x.second.x << "||" << x.second.y << "||" */
/*        << x.second.xId << "||" << x.second.yId << "||" << x.second.cost << "
 * "; */
/*   } */
/*   os << std::endl; */
/*   return os; */
/* } */
