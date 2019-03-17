import numpy as np
from physics_sim import PhysicsSim

 
class TakeOff():
    """Task (environment) that defines the goal and provides feedback to the agent."""

    def __init__(self, init_pose=np.array([0.0, 0.0, 1, 0.0, 0.0, 0.0]), init_velocities=None,
                 init_angle_velocities=None, runtime=5, target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities,
                              init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # how long have you stayed up
        self.step_count = 0
        
        
        # Goal
        self.target_pos = np.array([0., 0., 0.])

        # height we should reward
        self.minimum_z = 5
        

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = self.step_count * self.sim.pose[2] - .1 * (abs(self.sim.pose[0] - self.target_pos[0]) +
                                                                  abs(self.sim.pose[1] - self.target_pos[1]))
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        self.step_count += 1
        for _ in range(self.action_repeat):
            # update the sim pose and velocities
            done = self.sim.next_timestep(rotor_speeds)
            reward += self.get_reward()
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        if done:
            self.final_position = self.sim.pose[:3]
            if self.sim.pose[2] > self.minimum_z:
                reward += 10
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        self.step_count = 0
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state
