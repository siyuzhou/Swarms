# Swarms
Models of artificial swarms.

## Data generation

```bash
python obstacle_avoidance_sim.py --boids B --vicseks V --obstacles OBST --steps N_STEPS --save-dir /path/to/save/location/
```
B: No. of Boids  
V: No. of Vicseks  
OBST: No. of obstacles  
N_STEPS: No. of steps for which data is generated  

### Files Generated
*_edge.npy: Edge type data   
*_time.npy: Time steps(dt)  
*_timeseries.npy: N_STEPS x N x D tensor  

N: No. of Boids + No. of Vicseks + No. of obstacles  
D: State vector in form of (position, velocity)
## Boid

A simple implementation of Craig Reynolds' [Boids](https://www.red3d.com/cwr/boids/) model.  

![2D Flocking w/ Goal and Obstacle](demo/boid_goal_obstacle.gif)
![2D Flocking w/ Goal and Obstacle2](demo/boid_goal_obstacle2.gif)

## Vicsek

An implementation of [Helbing - Farkas - Vicsek](https://www.nature.com/articles/35035023) model. 

![2D Flocking w/ Goal and Obstacle](demo/vicsek_goal_obstacle.gif)