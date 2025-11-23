# lets import the packages we need
import gymnasium as gym
import torch 
import torch.nn as nn 
from torch.optim import AdamW
from torch.distributions import Categorical

import platform
import sys
import os

def is_macos():
    return sys.platform == "darwin" or platform.system() == "Darwin"

def is_cuda():
    return torch.cuda.is_available()

# Create the policy/agent
class PolicyModel(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        """
        state_dim: the size of the environment state
        action_dim: the number of possible actions the agent can take
        hidden_dim: the hidden size of the layers
        """
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        ) 

    def forward(self, state):
        """
        Args:
            state: the current state of the environment
        Return: The action distribution predicted by the model
        """

        pred_actions = self.model(state)

        return pred_actions
    
class Trainer():
    def __init__(self, env_name="CartPole-v1"):
        
        self.env_name = env_name
        self.env = gym.make(self.env_name)
        
        self.device = "cpu"
        if is_cuda():
            self.device = "cuda"
        elif is_macos():
            self.device = "mps"

        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n

        self.policy = PolicyModel(state_dim=state_dim, action_dim=action_dim, hidden_dim=128)
        self.policy.to(self.device)

        self.optimizer = AdamW(self.policy.parameters(), lr=1e-2)
    
    def save_weights(self, save_path: str):
        torch.save(self.policy.state_dict(), save_path)
    
    def load_weights(self, load_path: str):
        weights = torch.load(load_path)
        self.policy.load_state_dict(weights)

    def train(self, num_episodes: int, gamma:float = 0.99, save_dir: str = "out"):

        os.makedirs(save_dir, exist_ok=True)

        max_reward = 0.0
        for episode in range(num_episodes):

            log_probs = []
            rewards = []

            # reset the environment to starting point in each episode
            state, info = self.env.reset()

            done = False 

            while not done:

                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
                pred_action_probs = self.policy(state_tensor) # ex. [0.3, 0.7]

                pred_action_dist = Categorical(pred_action_probs)

                pred_action = pred_action_dist.sample()
                pred_action_prob = pred_action_dist.log_prob(pred_action)

                # store the log prob of the action
                log_probs.append(pred_action_prob)

                # take the action in the env
                next_state, reward, terminated, truncated, info = self.env.step(pred_action.item())
                #if render:
                    #self.env.render()

                rewards.append(reward)

                # update current state
                state = next_state

                done = terminated or truncated
            
            returns = []

            G = 0.0

            # compute returns by reverse transversing the list
            for r in reversed(rewards):
                G = r + gamma * G
                returns.insert(0, G)

            returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

            # compute loss

            loss = 0.0

            for log_prob, Gt in zip(log_probs, returns):

                loss += -log_prob * Gt

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # log rewards
            total_reward = sum(rewards)

            if total_reward > max_reward:
                max_reward = total_reward
                self.save_weights(f"{save_dir}/episode_{episode+1}.pth")
                print(f"New best reward {max_reward} at episode: {episode+1}")

            if (episode + 1) % 10 == 0:
                print(f"Episode: {episode + 1} total reward {total_reward:.1f}")

            

        self.env.close()

    def test(self):

        test_env = gym.make(self.env_name, render_mode="human")

        state, info = test_env.reset()

        rewards = 0.0

        done = False

        while not done:
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            pred_actions = self.policy(state)
            pred_action = pred_actions.argmax(dim=-1)

            next_state, reward, terminated, truncated, info = test_env.step(pred_action.item())

            rewards += reward
            state = next_state

            done = terminated or truncated

        print(f"Total rewards (test): {rewards}")


if __name__ == "__main__":

    trainer = Trainer()
    #trainer.train(200)

    # uncomment to test
    trainer.load_weights("out/episode_111.pth")
    trainer.test()


            




        