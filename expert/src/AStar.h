#ifndef ASTAR_H
#define ASTAR_H

#include <iostream>
#include <unordered_map>
#include <vector>

#pragma once

struct World {
  std::pair<float, float> X;
  std::pair<float, float> Y;
  std::pair<int, int> width;
};

struct Node {
  int xId, yId;
  float x, y;
  float cost;
  int parentIdx;
};

class AStar {
private:
  float resolution;
  struct World worldMap;
  float radius;
  std::vector<std::vector<float>> motionVector;
  std::vector<std::vector<bool>> obsMap;

public:
  AStar(float, float, struct World);
  ~AStar();
  void calcObsMap(float, float);
  float calcGridPosition(int, float);
  int calcGridIndex(struct Node);
  int calcXYIndex(float, float);
  void plan(float, float, float, float);
  float calcHeuristic(struct Node, struct Node);
  bool checkNode(struct Node);
  void calcPath(struct Node, std::unordered_map<int, struct Node>);
  void printPath();

  std::vector<std::pair<float, float>> path;

  /* friend std::ostream &operator<<(const std::ostream, */
  /*                                 const std::unordered_map<int, struct
   * Node>); */

  friend std::ostream &
  operator<<(std::ostream &os,
             const std::unordered_map<int, struct Node> &map) {
    for (auto x : map) {
      os << x.first << " " << x.second.x << "||" << x.second.y << "||"
         << x.second.xId << "||" << x.second.yId << "||" << x.second.cost
         << " ";
    }
    os << std::endl;
    return os;
  }

private:
};

#endif
