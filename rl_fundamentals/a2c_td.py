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

# Create the actor network
class ActorModel(nn.Module):
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

    def forward(self, state: torch.Tensor):
        """
        Args:
            state: the current state of the environment
        Return: The action logits predicted by the model
        """
    
        action_probs = self.model(state)
       
        return action_probs

# Create the actor network
class CriticModel(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        """
        state_dim: the size of the environment state
        hidden_dim: the hidden size of the layers
        """
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ) 

    def forward(self, state: torch.Tensor):
        """
        Args:
            state: the current state of the environment
        Return: The value estimated by the model
        """
    
        value = self.model(state).squeeze(-1)
       
        return value  

    
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

        self.policy = ActorModel(state_dim=state_dim, action_dim=action_dim, hidden_dim=128)
        self.policy.to(self.device)

        self.critic = CriticModel(state_dim=state_dim, hidden_dim=128)
        self.critic.to(self.device)

        self.policy_optimizer = AdamW(self.policy.parameters(), lr=1e-3)
        self.critic_optimizer = AdamW(self.critic.parameters(), lr=1e-2)
    
    def save_weights(self, save_path: str):
        torch.save(self.policy.state_dict(), save_path)
    
    def load_weights(self, load_path: str):
        weights = torch.load(load_path)
        self.policy.load_state_dict(weights)

    def train(
        self,
        num_episodes: int,
        gamma: float = 0.99,
        entropy_coeff: float = 0.01,
        save_dir: str = "out",
    ) -> str:

        os.makedirs(save_dir, exist_ok=True)

        max_reward = 0.0
        best_checkpoint = ""
        for episode in range(num_episodes):

            self.policy.train()

            rewards = []

            # reset the environment to starting point in each episode
            state, info = self.env.reset()

            done = False 

            while not done:


                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
                pred_action_probs = self.policy(state_tensor)

                pred_action_dist = Categorical(pred_action_probs)

                pred_action = pred_action_dist.sample()
                pred_action_log_probs = pred_action_dist.log_prob(pred_action)
                entropy = pred_action_dist.entropy()

                state_value = self.critic(state_tensor)

                # take the action in the env
                next_state, reward, terminated, truncated, info = self.env.step(pred_action.item())

                done = terminated or truncated

                with torch.no_grad():

                    if terminated:
                        next_state_value = torch.tensor(0.0, device=self.device)
                    else:
                        next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=self.device)
                        next_state_value = self.critic(next_state_tensor)

                
                # compute td error
                td_error = reward + gamma * next_state_value - state_value

                # compute actor loss
                actor_loss = -pred_action_log_probs * td_error.detach() - entropy_coeff * entropy

                actor_loss.backward()
                self.policy_optimizer.step()
                self.policy_optimizer.zero_grad()

                # compute critic loss
                state_value = self.critic(state_tensor)

                critic_loss = 0.5 * torch.pow(reward + gamma * next_state_value - state_value, 2)

                critic_loss.backward()
                self.critic_optimizer.step()

                self.critic_optimizer.zero_grad()

                state = next_state

                rewards.append(reward)

            # log rewards
            total_reward = sum(rewards)

            if total_reward > max_reward:
                max_reward = total_reward
                best_checkpoint = f"{save_dir}/episode_{episode+1}.pth"

                self.save_weights(best_checkpoint)

                print(f"New best reward {max_reward} at episode: {episode+1}")

            if (episode + 1) % 10 == 0:
                print(f"Episode: {episode + 1} total reward {total_reward:.1f}")

            

        self.env.close()

        return best_checkpoint

    def test(self):

        self.policy.eval()

        test_env = gym.make(self.env_name, render_mode="human")

        # get the initial state
        state, info = test_env.reset()

        rewards = 0.0

        done = False

        with torch.no_grad():

            while not done:

                state = torch.tensor(state, dtype=torch.float32, device=self.device)

                # get the action predictions
                pred_actions = self.policy(state)

                # take the action with the highest probability
                pred_action = pred_actions.argmax(dim=-1)

                # take the action in the env
                next_state, reward, terminated, truncated, info = test_env.step(pred_action.item())

                rewards += reward
                state = next_state

                done = terminated or truncated

            print(f"Total rewards (test): {rewards}")

        
        self.env.close()


if __name__ == "__main__":

    trainer = Trainer()
    output_dir = "./out5"

    #comment this to test an existing checkpoint
    #best_checkpoint = trainer.train(200, save_dir=output_dir)
    #comment this to test an existing checkpoint
    #trainer.load_weights(best_checkpoint)

    # uncomment to test exiting checkpoint
    trainer.load_weights(f"{output_dir}/episode_105.pth")
    trainer.test()




        
