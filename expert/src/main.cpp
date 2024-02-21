#include "AStar.h"
#include "matplotlibcpp.h"
#include <iostream>

namespace plt = matplotlibcpp;

int main(int, char **) {
  struct World map = {std::make_pair(-4., 4.), std::make_pair(-4., 4.),
                      std::make_pair(0, 0)};
  float radius = 0.13;
  float resolution = 0.1;

  AStar aStar(resolution, radius, map);

  std::vector<std::vector<float>> obstacles = {
      {1, 0.7},   {1, 0.8},   {1, 0.9},   {1, 1},     {1, 1.1},   {1, 1.2},
      {1, 1.3},   {1, 1.4},   {1, 1.5},   {1, 1.6},   {1, 1.7},   {1, 1.8},
      {2, 2},     {2.1, 2},   {2.2, 2},   {2.3, 2},   {2.4, 2},   {2.5, 2},
      {2.5, 1.9}, {2.5, 1.8}, {2.5, 1.7}, {2.5, 1.6}, {2.5, 1.5}, {2.8, 3.2},
      {2.9, 3.1}, {3, 3},     {3.1, 2.9}, {3.2, 2.8},
  };

  for (auto obstacle : obstacles) {
    aStar.calcObsMap(obstacle[0], obstacle[1]);
  }

  float sx = 2.3;
  float sy = 1.5;
  float gx = 2.8;
  float gy = 2.3;
  aStar.plan(sx, sy, gx, gy);

  aStar.printPath();

  /* plt::plot({1, 3, 2, 4}); */
  std::vector<float> x;
  std::vector<float> y;
  for (auto obstacle : obstacles) {
    x.push_back(obstacle[0]);
    y.push_back(obstacle[1]);
  }

  std::vector<float> pathX;
  std::vector<float> pathY;
  for (auto path : aStar.path) {
    pathX.push_back(path.first);
    pathY.push_back(path.second);
  }
  /* plt::scatter({0.}, {0.}); */
  /* plt::scatter({3.5}, {3.5}); */
  std::vector<float> ox = {sx, gx};
  std::vector<float> oy = {sy, gy};
  plt::scatter(ox, oy, 'g');
  plt::scatter(x, y, 'b');
  plt::plot(pathX, pathY, "r--");
  plt::show();
}
