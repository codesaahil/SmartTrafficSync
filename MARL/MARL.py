import gym
import numpy as np
import random

class TrafficLightEnv(gym.Env):
    def __init__(self):
        super(TrafficLightEnv, self).__init__()
        
        # Define action space (3 actions: Yellow, Green, Red)
        self.action_space = gym.spaces.Discrete(3)
        
        # Define observation space (state based on the number of vehicles in NS and EW)
        self.observation_space = gym.spaces.MultiDiscrete([21, 21])  # Vehicle count from 0 to 20 in each direction
        
        # Initialize the environment state
        self.state = None

    def reset(self):
        # Randomly initialize the number of vehicles in NS and EW
        self.state = (random.randint(0, 20), random.randint(0, 20))
        return self.state
    
    def get_state(self, vehicles_ns, vehicles_ew):
        # Return the state based on the number of vehicles in each direction
        return (vehicles_ns, vehicles_ew)
    
    def compute_reward(self, vehicles_ns, vehicles_ew, action):
        """
        Enhanced reward function:
        - Reward based on number of vehicles passing through the intersection
        - Reward for clearing the direction with more vehicles
        """
        if action == 1:  # Green (NS goes)
            reward = min(vehicles_ns, 5)  # Cap the reward to 5 vehicles passing
        elif action == 2:  # Red (EW goes)
            reward = min(vehicles_ew, 5)  # Cap the reward to 5 vehicles passing
        else:  # Yellow (stopping traffic)
            reward = -5  # Increased penalty for staying yellow too long
        
        # If NS has more vehicles, it should ideally be rewarded more for clearing NS, and vice versa
        if vehicles_ns > vehicles_ew:
            reward += 2 if action == 1 else -1  # Reward NS if it's cleared, penalize if not
        elif vehicles_ew > vehicles_ns:
            reward += 2 if action == 2 else -1  # Reward EW if it's cleared, penalize if not

        # Penalize if traffic is too congested in one direction
        if vehicles_ns > 15 or vehicles_ew > 15:
            reward -= (vehicles_ns - 15) if vehicles_ns > 15 else 0
            reward -= (vehicles_ew - 15) if vehicles_ew > 15 else 0
        
        # Reward for clearing all vehicles
        if vehicles_ns == 0 and vehicles_ew == 0:
            reward += 3  # Bonus for clearing all vehicles
        
        return reward
    
    def step(self, action):
        # Decode action (Yellow, Green, Red)
        if action == 0:  # Yellow
            self.state = self.get_state(self.state[0], self.state[1])
        elif action == 1:  # Green (NS goes)
            vehicles_to_clear = min(self.state[0], max(1, self.state[0] // 3))  # Dynamic reduction
            self.state = self.get_state(self.state[0] - vehicles_to_clear, self.state[1])  # Reduce NS vehicles
        elif action == 2:  # Red (EW goes)
            vehicles_to_clear = min(self.state[1], max(1, self.state[1] // 3))  # Dynamic reduction
            self.state = self.get_state(self.state[0], self.state[1] - vehicles_to_clear)  # Reduce EW vehicles
        
        vehicles_ns, vehicles_ew = self.state
        
        # Calculate reward
        reward = self.compute_reward(vehicles_ns, vehicles_ew, action)
        
        # In a real environment, you might have some terminal condition
        done = False  # Keep the simulation running indefinitely
        
        return self.state, reward, done, {}
    
    def render(self):
        pass  # Optional: implement to display the environment


class TrafficLightAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995):
        self.env = env
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Initial exploration rate
        self.epsilon_min = epsilon_min  # Minimum exploration rate
        self.epsilon_decay = epsilon_decay  # Epsilon decay factor
        
        # Initialize Q-table (state space x action space)
        self.q_table = np.zeros((21, 21, 3))  # 21x21 state space (vehicles in NS and EW), 3 possible actions
    
    def get_action(self, state):
        # Epsilon-greedy policy with epsilon decay
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()  # Explore: random action
        else:
            ns, ew = state
            return np.argmax(self.q_table[ns, ew])  # Exploit: best action based on Q-values
    
    def update_q_value(self, state, action, reward, next_state):
        ns, ew = state
        next_ns, next_ew = next_state
        best_next_action = np.argmax(self.q_table[next_ns, next_ew])
        self.q_table[ns, ew, action] = self.q_table[ns, ew, action] + self.alpha * (
            reward + self.gamma * self.q_table[next_ns, next_ew, best_next_action] - self.q_table[ns, ew, action]
        )
    
    def train(self, num_episodes=1000, max_steps=100):
        for episode in range(num_episodes):
            state = self.env.reset()
            
            for _ in range(max_steps):
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                
                # Update Q-table
                self.update_q_value(state, action, reward, next_state)
                
                # Transition to the next state
                state = next_state
                
            # Decay epsilon after each episode
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            if episode % 100 == 0:
                print(f"Episode {episode}/{num_episodes} completed with epsilon {self.epsilon:.3f}")

    def get_best_action(self, state):
        ns, ew = state
        return np.argmax(self.q_table[ns, ew])


# Initialize environment and agent
env = TrafficLightEnv()
agent = TrafficLightAgent(env)

# Train the agent using Q-learning
agent.train(num_episodes=10000)

# Test the trained agent
test_vehicles_ns = 9
test_vehicles_ew = 7
state = (test_vehicles_ns, test_vehicles_ew)
best_action = agent.get_best_action(state)

action_mapping = {0: 'Yellow', 1: 'Green (NS)', 2: 'Red (NS)'}
print(f"Best action for {test_vehicles_ns} vehicles in NS and {test_vehicles_ew} vehicles in EW is: {action_mapping[best_action]}")
