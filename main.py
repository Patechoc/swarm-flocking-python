import numpy as np 
from cv2 import cv2

class Agent():
    def __init__(self, init_position, init_velocity, scan_radius=25, max_vel=3, randomness=2, h_bound=[0,640], v_bound=[0,480]):
        self.position = init_position # Numpy array [x, y]
        self.velocity = init_velocity  # Numpy array [x, y]
        self.scan_radius = scan_radius # Int
        self.max_vel = max_vel
        self.randomness = randomness
        self.h_bound = h_bound
        self.v_bound = v_bound

    def update(self, env):
        new_velocity = np.array([0, 0], dtype=np.float)
        obs_in_rad = 0
        for object in env:
            distance = self.position - object.position
            dist_mag = np.linalg.norm(distance)
            if dist_mag <= self.scan_radius:
                obs_in_rad += 1
                # Calculate velocity vector
                if dist_mag < self.scan_radius/1.5:
                    new_velocity += distance * 100 / (dist_mag + 0.0001)
                else:
                    new_velocity += object.velocity
        
        # Taking average of total velocity
        new_velocity /= obs_in_rad
        # Adding new velocity to previous
        self.velocity += new_velocity
        # Adding random noise
        self.velocity += (np.random.rand(2) - 0.5) * self.randomness

        # Capping to max velocity
        if np.linalg.norm(self.velocity) > self.max_vel:
            self.velocity /= np.linalg.norm(self.velocity)
            self.velocity *= self.max_vel

        # Calculating new position
        new_position = self.position + self.velocity

        if new_position[0] < self.h_bound[0] or new_position[0] > self.h_bound[1]:
            new_position[0] = self.position[0]
            self.velocity[0] = 0
        if new_position[1] < self.v_bound[0] or new_position[1] > self.v_bound[1]:
            new_position[1] = self.position[1]
            self.velocity[1] = 0

        # Updating position
        self.position = new_position

    def draw(self, image):
        cv2.circle(image, (int(self.position[0]), int(self.position[1])), 5, (0,0,255), -1, cv2.LINE_AA)

if __name__ == "__main__":
    display = np.ones((480,640,3), dtype=np.uint8) * 255

    NUMBER_OF_AGENTS = 80

    # Populating agents
    agents = []
    for i in range(NUMBER_OF_AGENTS):
        agents.append(Agent(np.random.rand(2)*display.shape[0], 
                            np.random.rand(2) - 0.5))

    while True:
        for agent in agents:
            agent.draw(display)
            agent.update(agents)

        cv2.imshow('sim', display)
        cv2.waitKey(15)
        display[:,:,:] = 255