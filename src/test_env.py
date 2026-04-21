from football_env import AIFootballEnv
import time

def run_test():
    print("Initializing AI Football Environment...")
    env = AIFootballEnv()
    
    # Start the game
    observations, infos = env.reset()
    print("✅ Environment loaded successfully!")
    print(f"Agents detected: {env.agents}\n")
    
    # Run the simulation for 60 frames (1 second of game time)
    for step in range(60):
        # 1. Generate random actions for each agent (acting like a confused AI)
        actions = {}
        for agent in env.agents:
            # .sample() generates a random array of 4 numbers between -1.0 and 1.0
            actions[agent] = env.action_space(agent).sample()
            
        # 2. Step the physics engine forward
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # 3. Print the rewards to the console to verify our math
        print(f"Frame {step} Rewards:")
        print(f"  🥅 Пандевалдо (GK):  {rewards['gk']:.3f}")
        print(f"  🛡️ Панчевалдо (DEF): {rewards['def']:.3f}")
        print(f"  ⚔️ Елмасалдо (ATT):  {rewards['att']:.3f}")
        print("-" * 30)
        
        # If the environment finishes (e.g., time runs out), break
        if not env.agents:
            print("Game Over!")
            break

if __name__ == "__main__":
    run_test()