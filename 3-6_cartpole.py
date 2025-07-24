"""Section 3.6 Experimental Environment.
"""

import gym

# Create the CartPole environment with human rendering
env = gym.make("CartPole-v0", render_mode="human")
state = env.reset()

# Run the environment for 1000 timesteps
for t in range(100):
    env.render()  # Render the environment
    #print(state)  # Print the current state

    action = env.action_space.sample()  # Sample a random action

    # * Take the action and get the next state and info
    # state: the next observation of the environment
    # reward: the reward received after taking the action
    # terminated: whether the episode has ended (success or failure)
    # truncated: whether the episode was truncated (e.g., time limit reached)
    # info: additional diagnostic information
    state, reward, terminated, truncated, info = env.step(action)
    print(action)
    # Check if the episode is finished
    if terminated or truncated:
        print("Finished")
        state = env.reset()  # Reset the environment


env.close()  # Close the environment
"""
git test
"""