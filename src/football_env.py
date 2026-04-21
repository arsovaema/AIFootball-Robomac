import functools
import numpy as np
from pettingzoo import ParallelEnv
from gymnasium import spaces

# Import the physics objects from your existing game file
from AIFootball import Player, Ball, dt, resolve_collision, initial_positions_team_left

class AIFootballEnv(ParallelEnv):
    metadata = {
        "name": "aifootball_v0",
    }

    def __init__(self):
        # 1. Define our specific independent agents
        self.possible_agents = ["gk", "def", "att"]
        self.agents = self.possible_agents[:]

        # Define maximums based on your team_properties()
        # For mapping neural network outputs (-1 to 1) to actual game values
        self.max_force = 200 * 0.75  # Example based on your AIFootball.py coefficients
        self.max_shot_power = 200 * 0.95

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # The AI needs to see the game state. 
        # Let's say we give it 14 numbers: 
        # [self.x, self.y, self.v, self.alpha, ball.x, ball.y, ball.v, ball.alpha, + teammates/opponents]
        # We normalize everything between -1.0 and 1.0 so the neural network learns faster.
        return spaces.Box(low=-1.0, high=1.0, shape=(14,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        # The AI will output 4 continuous numbers between -1.0 and 1.0.
        # We will translate these to: [force, alpha, shot_power, shot_request]
        return spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.timestep = 0

        # Initialize the physics objects from your AIFootball.py
        # Here we pass the specific weight/radius points you mentioned earlier
        self.our_team = {
            "gk": Player("Nate", weight=9, radius=20, acceleration=40, speed=40, shot_power=18),
            "def": Player("Cassie", weight=10, radius=10, acceleration=10, speed=10, shot_power=20),
            "att": Player("Maddie", weight=15, radius=5, acceleration=15, speed=25, shot_power=13)
        }
        
        # Reset positions to standard starting spots
        self.our_team["gk"].reset(initial_positions_team_left[0], 0)
        self.our_team["def"].reset(initial_positions_team_left[1], 0)
        self.our_team["att"].reset(initial_positions_team_left[2], 0)

        self.ball = Ball()
        self.ball.reset()

        # Get initial observations for all agents
        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos

    def step(self, actions):
        self.timestep += 1

        # 1. APPLY ACTIONS
        for agent_id, action_array in actions.items():
            player = self.our_team[agent_id]
            
            # Map the Neural Network output (-1.0 to 1.0) back to your manager.py dictionary format
            decision = {
                'force': action_array[0] * self.max_force, # Maps to -max to +max
                'alpha': action_array[1] * np.pi,          # Maps to -pi to +pi
                'shot_power': abs(action_array[2] * self.max_shot_power), # 0 to max
                'shot_request': True if action_array[3] > 0 else False
            }
            
            # Execute your physics engine movement
            player.move(decision)

        # 2. STEP PHYSICS (Move the ball and resolve collisions)
        self.ball.move()
        self.ball.snelius()
        for player in self.our_team.values():
            player.snelius()
            if self._check_collision(player, self.ball):
                resolve_collision(player, self.ball)

        # 3. CALCULATE REWARDS
        # This is where we will shape the unique rewards for the GK, Def, and Attacker
        rewards = {
            "gk": self._calculate_gk_reward(),
            "def": self._calculate_def_reward(),
            "att": self._calculate_att_reward()
        }

        # 4. CHECK END STATE
        # E.g., Ends when a goal is scored or max time is reached
        env_done = self.timestep >= 3000 # Roughly 50 seconds at 60fps
        
        terminations = {agent: env_done for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        observations = {agent: self._get_obs(agent) for agent in self.agents}

        if env_done:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    # --- Helper Functions ---
    def _get_obs(self, agent_id):
        """
        Translates the game state into a 14-number array for the neural network.
        Everything is normalized between -1.0 and 1.0.
        """
        player = self.our_team[agent_id]
        
        # Game constants for normalization
        MAX_X = 1366.0
        MAX_Y = 768.0
        MAX_V = player.v_max if player.v_max > 0 else 1.0
        
        # 1. The Agent's own state (4 values)
        # We divide by max width/height to get a percentage, then shift to -1 to 1
        my_x = (player.x / MAX_X) * 2.0 - 1.0
        my_y = (player.y / MAX_Y) * 2.0 - 1.0
        my_v = (player.v / MAX_V) * 2.0 - 1.0
        my_alpha = player.alpha / np.pi # Alpha is already -pi to pi, so dividing by pi makes it -1 to 1
        
        # 2. The Ball's state (4 values)
        ball_x = (self.ball.x / MAX_X) * 2.0 - 1.0
        ball_y = (self.ball.y / MAX_Y) * 2.0 - 1.0
        ball_v = (self.ball.v / self.ball.v_max) * 2.0 - 1.0
        ball_alpha = self.ball.alpha / np.pi
        
        # 3. Teammate positions (4 values)
        # We gather the x, y of the OTHER two players on the team
        teammate_coords = []
        for other_id, other_player in self.our_team.items():
            if other_id != agent_id:
                teammate_coords.append((other_player.x / MAX_X) * 2.0 - 1.0)
                teammate_coords.append((other_player.y / MAX_Y) * 2.0 - 1.0)
                
        # 4. Target Goal Position (2 values)
        # Assuming we are training on the Left side, attacking the Right Goal (x=1316, y=460)
        goal_x = (1316.0 / MAX_X) * 2.0 - 1.0
        goal_y = (460.0 / MAX_Y) * 2.0 - 1.0

        # Combine them all into one flat numpy array (14 numbers total)
        observation = np.array([
            my_x, my_y, my_v, my_alpha,
            ball_x, ball_y, ball_v, ball_alpha,
            teammate_coords[0], teammate_coords[1], teammate_coords[2], teammate_coords[3],
            goal_x, goal_y
        ], dtype=np.float32)

        return observation

    def _check_collision(self, circle_1, circle_2):
        # Fast math to check if two circles are overlapping
        dist_sq = (circle_1.x - circle_2.x)**2 + (circle_1.y - circle_2.y)**2
        radius_sum_sq = (circle_1.radius + circle_2.radius)**2
        return dist_sq <= radius_sum_sq

    def _calculate_gk_reward(self):
        """
        Пандевалдо (Goalkeeper): 
        Wants to stay near the left goal (x=50, y=460) and keep the ball away from it.
        """
        player = self.our_team["gk"]
        reward = 0.0
        
        # 1. Positional Reward: Stay near the goal!
        dist_to_goal = np.sqrt((50 - player.x)**2 + (460 - player.y)**2)
        if dist_to_goal < 150:
            reward += 0.1  # Good boy, staying in the box
        else:
            reward -= (dist_to_goal / 1000) # Penalty for wandering off
            
        # 2. Save Reward: Touching the ball near the goal is a save
        dist_to_ball = np.sqrt((self.ball.x - player.x)**2 + (self.ball.y - player.y)**2)
        if dist_to_ball < 30 and player.x < 300: 
            reward += 5.0 # Massive reward for a block!
            
        # 3. Concede Penalty (The Stick)
        if self.ball.x < 50 and 343 < self.ball.y < 578:
            reward -= 100.0 # Huge penalty for letting a goal in

        return reward

    def _calculate_def_reward(self):
        """
        Панчевалдо (Defender):
        Wants to intercept the ball in the middle and pass it forward.
        """
        player = self.our_team["def"]
        reward = 0.0
        
        # 1. Positional Penalty: Don't go too far forward!
        if player.x > 800:
            reward -= 0.5 # Get back on defense!
            
        # 2. Interception Reward: Get close to the ball
        dist_to_ball = np.sqrt((self.ball.x - player.x)**2 + (self.ball.y - player.y)**2)
        # Closer is better, mapped so 0 distance = +1 reward
        reward += max(0, 1.0 - (dist_to_ball / 500))
        
        # 3. Clearance Reward: Pushing the ball towards the opponent's half
        if self.ball.v > 10 and self.ball.x > 683: # 683 is midfield
            reward += 0.5 

        return reward

    def _calculate_att_reward(self):
        """
        Елмасалдо (Attacker):
        Wants to get the ball into the right goal (x=1316, y=460) at all costs.
        """
        player = self.our_team["att"]
        reward = 0.0
        
        # 1. Positional Penalty: Don't hang out in our own penalty box
        if player.x < 300:
            reward -= 0.5 # Stop playing defense!
            
        # 2. Ball Control: Get to the ball
        dist_to_ball = np.sqrt((self.ball.x - player.x)**2 + (self.ball.y - player.y)**2)
        reward += max(0, 1.0 - (dist_to_ball / 300))
        
        # 3. Offensive Pressure: Ball moving toward the enemy goal
        dist_ball_to_enemy_goal = np.sqrt((1316 - self.ball.x)**2 + (460 - self.ball.y)**2)
        # The smaller the distance, the higher the reward
        reward += max(0, 5.0 - (dist_ball_to_enemy_goal / 200))
        
        # 4. GOAL!!! (The Ultimate Carrot)
        if self.ball.x > 1316 and 343 < self.ball.y < 578:
            reward += 100.0 # Massive reward for scoring

        return reward