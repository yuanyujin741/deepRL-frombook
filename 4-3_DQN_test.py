from RL_utils import *
import gym
device = "cuda" if torch.cuda.is_available() else "cpu"

env = gym.make("CartPole-v1")
config = Config()
model = SimpleNN()
# evaluate the model
evaluate_time = 10
episodes = [0, 100, 200, 300, 400, 500, 600,700,800,900,1000]
for episode_num in episodes:
    model_reward = []
    model.load_state_dict(torch.load(f"2model_trained/dqn_model_episode{episode_num}.pth"))
    for i in range(evaluate_time):
        state, info = env.reset()
        episode_reward = 0
        while True:
            action = select_action(state=state,env=env,model=model,epsilon=Epsilon(0,0,0),config=config,device=device)
            state, reward, done, _, info = env.step(action)
            if done:
                reward = -5
            episode_reward += reward
            if done:
                break
        model_reward.append(episode_reward)
    print(f"episode:{episode_num}, reward:{model_reward}, average_reward:{sum(model_reward)/evaluate_time}")