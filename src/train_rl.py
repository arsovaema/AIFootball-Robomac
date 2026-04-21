import os
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
# Importing your specific environment
from football_env import AIFootballEnv 

def train():
    # 1. Create the environment
    # We use render_mode=None during training to make it run 10x faster
    env = AIFootballEnv()

    # 2. The "Translator" (SuperSuit)
    # This wraps your 3-player team so they share one neural network (Parameter Sharing)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    
    # 3. Vectorize for Stable Baselines 3
    # This makes the environment compatible with the PPO algorithm
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class='stable_baselines3')

    # 4. Initialize the Algorithm (PPO)
    # PPO is the standard for games with movement and physics
    model = PPO(
        policy="MlpPolicy", 
        env=env, 
        verbose=1, 
        learning_rate=1e-4, 
        tensorboard_log="./logs/football_ppo/"
    )

    # 5. Setup Auto-Saving
    # Saves a backup of the "brain" every 20,000 steps
    checkpoint = CheckpointCallback(save_freq=20000, save_path="./models/", name_prefix="rl_player")

    # 6. Start Training
    print("Training is starting... your agents are learning to play!")
    model.learn(total_timesteps=200000, callback=checkpoint)

    # 7. Save the final result
    model.save("aifootball_final_model")
    print("Success! Final model saved as aifootball_final_model.zip")

if __name__ == "__main__":
    train()