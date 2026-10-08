// N100 adapter to unmodified Apache-2.0 RVO2; all agents remain present.
#include "RVO.h"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

int main() {
  int n, steps;
  float dt, radius, maxspeed, horizon;
  if (!(std::cin >> n >> steps >> dt >> radius >> maxspeed >> horizon)) return 2;
  RVO::RVOSimulator sim(dt, 3.0F, n-1, horizon, horizon, radius, maxspeed);
  std::vector<RVO::Vector2> goals;
  for (int i=0;i<n;++i) {
    float sx,sy,gx,gy; std::cin >> sx >> sy >> gx >> gy;
    sim.addAgent(RVO::Vector2(sx,sy)); goals.push_back(RVO::Vector2(gx,gy));
  }
  // Clockwise enclosing polygon presents its free side to interior agents.
  std::vector<RVO::Vector2> wall;
  wall.push_back(RVO::Vector2(-1,-1)); wall.push_back(RVO::Vector2(-1,1));
  wall.push_back(RVO::Vector2(1,1)); wall.push_back(RVO::Vector2(1,-1));
  sim.addObstacle(wall); sim.processObstacles();
  std::cout << std::setprecision(10);
  for (int k=0;k<=steps;++k) {
    std::cout << k*static_cast<double>(dt);
    for (int i=0;i<n;++i) {
      auto p=sim.getAgentPosition(i); auto v=sim.getAgentVelocity(i);
      std::cout << ' ' << p.x() << ' ' << p.y() << ' ' << v.x() << ' ' << v.y();
    }
    std::cout << '\n';
    if(k==steps) break;
    for(int i=0;i<n;++i) {
      auto d=goals[i]-sim.getAgentPosition(i);
      float norm=std::sqrt(RVO::absSq(d));
      auto v=d*(std::min(maxspeed,norm/0.25F)/std::max(norm,1.e-12F));
      // Fixed weak clockwise preference removes exact reciprocal symmetry.
      if(norm>0.02F) v=v+RVO::Vector2(d.y(),-d.x())*(0.01F/std::max(norm,1.e-12F));
      float speed=std::sqrt(RVO::absSq(v));
      if(speed>maxspeed) v=v*(maxspeed/speed);
      sim.setAgentPrefVelocity(i,v);
    }
    sim.doStep();
  }
}
