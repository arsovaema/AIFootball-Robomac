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

    def __init__(self, render_mode=None):
        # 1. Define our specific independent agents
        super().__init__()
        self.render_mode=render_mode
        self.possible_agents = ["gk", "def", "att"]
        self.agents = self.possible_agents[:]
        
        # Define maximums based on your team_properties()
        # For mapping neural network outputs (-1 to 1) to actual game values
        self.max_force = 200 * 0.75      # Example based on your AIFootball.py coefficients
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
        self.our_team = {
            "gk":  Player("Nate",   weight=9,  radius=20, acceleration=40, speed=40, shot_power=18),
            "def": Player("Cassie", weight=10, radius=10, acceleration=10, speed=10, shot_power=20),
            "att": Player("Maddie", weight=15, radius=5,  acceleration=15, speed=25, shot_power=13),
        }

        # Reset positions to standard starting spots
        self.our_team["gk"].reset(initial_positions_team_left[0], 0)
        self.our_team["def"].reset(initial_positions_team_left[1], 0)
        self.our_team["att"].reset(initial_positions_team_left[2], 0)

        self.ball = Ball()
        self.ball.reset()

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
                "force":        action_array[0] * self.max_force,
                "alpha":        action_array[1] * np.pi,
                "shot_power":   abs(action_array[2] * self.max_shot_power),
                "shot_request": True if action_array[3] > 0 else False,
            }

            player.move(decision)

        # 2. STEP PHYSICS
        self.ball.move()
        self.ball.snelius()
        for player in self.our_team.values():
            player.snelius()
            if self._check_collision(player, self.ball):
                resolve_collision(player, self.ball)

        # 3. CALCULATE REWARDS
        rewards = {
            "gk":  self._calculate_gk_reward(),
            "def": self._calculate_def_reward(),
            "att": self._calculate_att_reward(),
        }

        # 4. CHECK END STATE
        env_done = self.timestep >= 3000  # ~50 seconds at 60fps

        terminations = {agent: env_done for agent in self.agents}
        truncations  = {agent: False     for agent in self.agents}
        infos        = {agent: {}        for agent in self.agents}
        observations = {agent: self._get_obs(agent) for agent in self.agents}

        if env_done:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    # -------------------------------------------------------------------------
    # Helper: observation builder
    # -------------------------------------------------------------------------
    def _get_obs(self, agent_id):
        """
        Translates the game state into a 14-number array for the neural network.
        Everything is normalised between -1.0 and 1.0.
        """
        player = self.our_team[agent_id]

        MAX_X = 1366.0
        MAX_Y = 768.0
        MAX_V = player.v_max if player.v_max > 0 else 1.0

        # 1. Agent's own state (4 values)
        my_x    = (player.x / MAX_X) * 2.0 - 1.0
        my_y    = (player.y / MAX_Y) * 2.0 - 1.0
        my_v    = (player.v / MAX_V) * 2.0 - 1.0
        my_alpha = player.alpha / np.pi

        # 2. Ball's state (4 values)
        ball_x    = (self.ball.x / MAX_X) * 2.0 - 1.0
        ball_y    = (self.ball.y / MAX_Y) * 2.0 - 1.0
        ball_v    = (self.ball.v / self.ball.v_max) * 2.0 - 1.0
        ball_alpha = self.ball.alpha / np.pi

        # 3. Teammate positions (4 values)
        teammate_coords = []
        for other_id, other_player in self.our_team.items():
            if other_id != agent_id:
                teammate_coords.append((other_player.x / MAX_X) * 2.0 - 1.0)
                teammate_coords.append((other_player.y / MAX_Y) * 2.0 - 1.0)

        # 4. Target goal position (2 values) — attacking the right goal
        goal_x = (1316.0 / MAX_X) * 2.0 - 1.0
        goal_y = (460.0  / MAX_Y) * 2.0 - 1.0

        observation = np.array([
            my_x, my_y, my_v, my_alpha,
            ball_x, ball_y, ball_v, ball_alpha,
            teammate_coords[0], teammate_coords[1],
            teammate_coords[2], teammate_coords[3],
            goal_x, goal_y,
        ], dtype=np.float32)

        return observation

    # -------------------------------------------------------------------------
    # Helper: collision check
    # -------------------------------------------------------------------------
    def _check_collision(self, circle_1, circle_2):
        dist_sq = (circle_1.x - circle_2.x) ** 2 + (circle_1.y - circle_2.y) ** 2
        radius_sum_sq = (circle_1.radius + circle_2.radius) ** 2
        return dist_sq <= radius_sum_sq

    # -------------------------------------------------------------------------
    # Helper: possession weights
    # -------------------------------------------------------------------------
    def _get_possession_weights(self):
        """
        Returns (weights, distances) for each agent.

        weights: dict[agent_id -> float in (0, 1)], sum == 1.0
            The player closest to the ball gets a weight near 1.0;
            others taper off smoothly via a softmin over distances.

        distances: dict[agent_id -> float]
            Raw Euclidean distance from each player to the ball.

        Tuning:
            T (temperature) controls how sharply possession is awarded.
            Lower T  -> winner-takes-all (e.g. T=80)
            Higher T -> smoother three-way sharing (e.g. T=300)
        """
        T = 150.0

        distances = {
            agent_id: np.sqrt(
                (self.ball.x - player.x) ** 2 + (self.ball.y - player.y) ** 2
            )
            for agent_id, player in self.our_team.items()
        }

        # Softmin: closest player gets the highest exponential value
        inv   = {k: np.exp(-v / T) for k, v in distances.items()}
        total = sum(inv.values())
        weights = {k: v / total for k, v in inv.items()}

        return weights, distances

    # -------------------------------------------------------------------------
    # Rewards
    # -------------------------------------------------------------------------
    def _calculate_gk_reward(self):
        """
        Goalkeeper — Nate.

        Ball possession mode  (w ≈ 1): chase ball, make saves.
        Formation mode        (w ≈ 0): stay near the left goal mouth.
        Always: massive penalty for conceding.
        """
        player = self.our_team["gk"]
        weights, distances = self._get_possession_weights()
        w = weights["gk"]
        reward = 0.0

        # --- Ball possession mode ---
        ball_reward = 0.0
        dist_to_ball = distances["gk"]

        # Reward for closing down the ball (useful for clearing danger)
        ball_reward += max(0, 2.0 - (dist_to_ball / 150))

        # Extra reward for physically touching the ball in own half (a save / clearance)
        if dist_to_ball < 30 and player.x < 300:
            ball_reward += 5.0

        # --- Formation mode ---
        form_reward = 0.0
        dist_to_goal = np.sqrt((50 - player.x) ** 2 + (460 - player.y) ** 2)

        if dist_to_goal < 150:
            form_reward += 0.2          # Good: staying in the box
        else:
            form_reward -= dist_to_goal / 800   # Penalty for wandering

        # --- Blend by possession weight ---
        reward += w * ball_reward + (1 - w) * form_reward

        # --- Always-on penalty: conceding a goal ---
        if self.ball.x < 50 and 343 < self.ball.y < 578:
            reward -= 100.0

        return reward

    def _calculate_def_reward(self):
        """
        Defender — Cassie.

        Ball possession mode  (w ≈ 1): intercept and clear forward.
        Formation mode        (w ≈ 0): triangulate between GK and ATT to hold shape.
        Always: penalty for pushing too far forward.
        """
        player = self.our_team["def"]
        weights, distances = self._get_possession_weights()
        w = weights["def"]
        reward = 0.0

        # --- Ball possession mode ---
        ball_reward = 0.0
        dist_to_ball = distances["def"]

        # Reward for closing down the ball
        ball_reward += max(0, 1.5 - (dist_to_ball / 400))

        # Clearance bonus: ball moving into the opponent half after DEF interaction
        if self.ball.v > 10 and self.ball.x > 683:
            ball_reward += 1.0

        # --- Formation mode: stay between GK and ATT ---
        form_reward = 0.0
        gk  = self.our_team["gk"]
        att = self.our_team["att"]

        ideal_x = (gk.x + att.x) / 2.0
        ideal_y = (gk.y + att.y) / 2.0
        dist_to_ideal = np.sqrt(
            (ideal_x - player.x) ** 2 + (ideal_y - player.y) ** 2
        )
        form_reward += max(0, 1.0 - (dist_to_ideal / 300))

        # --- Always-on penalty: over-committing up the pitch ---
        if player.x > 800:
            reward -= 0.5

        # --- Blend ---
        reward += w * ball_reward + (1 - w) * form_reward

        return reward

    def _calculate_att_reward(self):
        """
        Attacker — Maddie.

        Ball possession mode  (w ≈ 1): dribble, shoot, score.
        Formation mode        (w ≈ 0): make runs into the attacking third.
        Always: penalty for sitting in own half.
        """
        player = self.our_team["att"]
        weights, distances = self._get_possession_weights()
        w = weights["att"]
        reward = 0.0

        # --- Ball possession mode ---
        ball_reward = 0.0
        dist_to_ball = distances["att"]

        # Reward for being on the ball
        ball_reward += max(0, 2.0 - (dist_to_ball / 250))

        # Reward for ball being close to the enemy goal
        dist_ball_to_goal = np.sqrt(
            (1316 - self.ball.x) ** 2 + (460 - self.ball.y) ** 2
        )
        ball_reward += max(0, 5.0 - (dist_ball_to_goal / 200))

        # GOAL!
        if self.ball.x > 1316 and 343 < self.ball.y < 578:
            ball_reward += 100.0

        # --- Formation mode: make dangerous runs ---
        form_reward = 0.0

        # Reward for being in the attacking third, ready to receive
        if player.x > 900:
            form_reward += 0.3

        # Reward for being central (more dangerous position)
        dist_to_centre_y = abs(player.y - 460)
        form_reward += max(0, 0.5 - (dist_to_centre_y / 300))

        # --- Always-on penalty: camping in own half ---
        if player.x < 300:
            reward -= 0.5

        # --- Blend ---
        reward += w * ball_reward + (1 - w) * form_reward

        return reward