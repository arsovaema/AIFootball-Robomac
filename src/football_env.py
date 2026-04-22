# football_env.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import functools
import numpy as np
from pettingzoo import ParallelEnv
from gymnasium import spaces
from AIFootball import Player, Ball, dt, resolve_collision, initial_positions_team_left

# ── Константи ──────────────────────────────────────────────────────────────
FIELD_W      = 1366.0
FIELD_H      = 768.0
GOAL_X_LEFT  = 50
GOAL_X_RIGHT = 1316
GOAL_TOP     = 343
GOAL_BOT     = 578
GOAL_CY      = (GOAL_TOP + GOAL_BOT) / 2

GK_X_LOCK  = 85
GK_Y_MIN   = GOAL_TOP
GK_Y_MAX   = GOAL_BOT
GK_GOAL_CY = GOAL_CY


class AIFootballEnv(ParallelEnv):
    metadata = {"name": "aifootball_v0"}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode     = render_mode
        self.possible_agents = ["gk", "def", "att"]
        self.agents          = self.possible_agents[:]
        self.max_force       = 200 * 0.75
        self.max_shot_power  = 200 * 0.95

    # ── Observation spaces — различни за секој агент ──────────────────────
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        if agent == "gk":
            return spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        elif agent == "def":
            return spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        else:  # att
            return spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)

    # ── Action spaces — различни за секој агент ───────────────────────────
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        if agent == "gk":
            return spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        elif agent == "def":
            return spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        else:  # att
            return spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

    # ── Reset ─────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        self.agents   = self.possible_agents[:]
        self.timestep = 0

        # FIX 6: flag за да се регистрира гол само еднаш
        self.goal_scored = False

        self.our_team = {
            "gk":  Player("Nate",   weight=20, radius=20, acceleration=40, speed=40, shot_power=30),
            "def": Player("Maddie", weight=15, radius=15, acceleration=10, speed=15, shot_power=25),
            "att": Player("Cassie", weight=15, radius=20, acceleration=15, speed=25, shot_power=18),
        }

        self.our_team["gk"].x     = GK_X_LOCK
        self.our_team["gk"].y     = GK_GOAL_CY
        self.our_team["gk"].alpha = 0
        self.our_team["gk"].v     = 0

        self.our_team["def"].reset(initial_positions_team_left[1], 0)
        self.our_team["att"].reset(initial_positions_team_left[2], 0)

        self.ball = Ball()
        self.ball.reset()

        # FIX 3: иницијализирај prev_dist_* веднаш во reset() за да не се добие
        # AttributeError при првиот повик на _reward_gk() / _reward_def()
        att  = self.our_team["att"]
        gk   = self.our_team["gk"]
        defp = self.our_team["def"]
        self.prev_dist_att_gk  = np.sqrt((att.x  - gk.x)**2   + (att.y  - gk.y)**2)
        self.prev_dist_def_gk  = np.sqrt((defp.x - gk.x)**2   + (defp.y - gk.y)**2)
        self.prev_dist_att_def = np.sqrt((att.x  - defp.x)**2  + (att.y  - defp.y)**2)

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos        = {agent: {}                   for agent in self.agents}
        return observations, infos

    # ── Step ──────────────────────────────────────────────────────────────
    def step(self, actions):
        self.timestep += 1

        # 1. ГОЛМАНОТ — само Y движење
        gk     = self.our_team["gk"]
        new_y  = np.clip(gk.y + actions["gk"][0] * gk.v_max * dt, GK_Y_MIN, GK_Y_MAX)
        gk.x   = GK_X_LOCK
        gk.y   = new_y
        gk.v   = 0

        # 2. DEF и ATT — полна контрола
        for agent_id in ["def", "att"]:
            action_array = actions[agent_id]
            player       = self.our_team[agent_id]
            decision     = {
                "force":        action_array[0] * self.max_force,
                "alpha":        action_array[1] * np.pi,
                "shot_power":   abs(action_array[2] * self.max_shot_power),
                "shot_request": action_array[3] > 0,
            }
            player.move(decision)

        # 3. PHYSICS
        self.ball.move()
        self.ball.snelius()
        for player in self.our_team.values():
            player.snelius()
            if self._check_collision(player, self.ball):
                resolve_collision(player, self.ball)

        # 4. Заклучи го голманот по колизии
        gk.x = GK_X_LOCK
        gk.y = np.clip(gk.y, GK_Y_MIN, GK_Y_MAX)
        gk.v = 0

        # Пресметај тековни дистанци
        att  = self.our_team["att"]
        gk   = self.our_team["gk"]
        defp = self.our_team["def"]
        curr_dist_att_gk  = np.sqrt((att.x  - gk.x)**2   + (att.y  - gk.y)**2)
        curr_dist_def_gk  = np.sqrt((defp.x - gk.x)**2   + (defp.y - gk.y)**2)
        curr_dist_att_def = np.sqrt((att.x  - defp.x)**2  + (att.y  - defp.y)**2)

        # Зачувај за rewards
        self.curr_dist_att_gk  = curr_dist_att_gk
        self.curr_dist_def_gk  = curr_dist_def_gk
        self.curr_dist_att_def = curr_dist_att_def

        # FIX 6: провери гол само ако уште не е регистриран
        just_scored_goal = False
        if not self.goal_scored:
            if (self.ball.x > GOAL_X_RIGHT and
    GOAL_TOP < self.ball.y < GOAL_BOT):
                self.goal_scored  = True
                just_scored_goal  = True

        # FIX 1+2: само едно место за rewards, без дупликат
        rewards = {
            "gk":  self._reward_gk(),
            "def": self._reward_def(just_scored_goal),
            "att": self._reward_att(just_scored_goal),
        }

        # Ажурирај претходни дистанци — само еднаш, на крај
        self.prev_dist_att_gk  = curr_dist_att_gk
        self.prev_dist_def_gk  = curr_dist_def_gk
        self.prev_dist_att_def = curr_dist_att_def

        # 6. END STATE
        env_done     = self.timestep >= 3000 or self.goal_scored
        terminations = {agent: env_done for agent in self.agents}
        truncations  = {agent: False    for agent in self.agents}
        infos        = {agent: {}       for agent in self.agents}
        observations = {agent: self._get_obs(agent) for agent in self.agents}

        if env_done:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    # ── Observations ──────────────────────────────────────────────────────
    def _get_obs(self, agent_id):
        player = self.our_team[agent_id]

        def nx(v): return (v / FIELD_W) * 2.0 - 1.0
        def ny(v): return (v / FIELD_H) * 2.0 - 1.0
        def nv(v, mx): return np.clip(v / max(mx, 1), 0, 1) * 2.0 - 1.0

        ball = self.ball

        if agent_id == "gk":
            dist = np.sqrt((ball.x - player.x)**2 + (ball.y - player.y)**2)
            return np.array([
                ny(player.y),
                nx(ball.x),
                ny(ball.y),
                nv(ball.v, ball.v_max),
                np.clip(dist / FIELD_W, 0, 1) * 2 - 1,
            ], dtype=np.float32)

        elif agent_id == "def":
            att = self.our_team["att"]
            return np.array([
                nx(player.x),  ny(player.y),
                nv(player.v, player.v_max),
                player.alpha / np.pi,
                nx(ball.x),    ny(ball.y),
                nv(ball.v, ball.v_max),
                ball.alpha / np.pi,
                nx(att.x),     ny(att.y),
                nx(GOAL_X_RIGHT), ny(GOAL_CY),
            ], dtype=np.float32)

        else:  # att
            defp = self.our_team["def"]
            return np.array([
                nx(player.x),  ny(player.y),
                nv(player.v, player.v_max),
                player.alpha / np.pi,
                nx(ball.x),    ny(ball.y),
                nv(ball.v, ball.v_max),
                ball.alpha / np.pi,
                nx(defp.x),    ny(defp.y),
                nx(GOAL_X_RIGHT), ny(GOAL_CY),
            ], dtype=np.float32)

    # ── Collision check ───────────────────────────────────────────────────
    def _check_collision(self, c1, c2):
        return (c1.x-c2.x)**2 + (c1.y-c2.y)**2 <= (c1.radius+c2.radius)**2

    def _proximity_penalty(self, agent_id):
        player  = self.our_team[agent_id]
        penalty = 0.0

        for other_id, other in self.our_team.items():
            if other_id == agent_id:
                continue
            dist     = np.sqrt((player.x - other.x)**2 + (player.y - other.y)**2)
            min_dist = player.radius + other.radius
            safe_dist = min_dist * 4

            if dist < safe_dist:
                penalty -= (1.0 - dist / safe_dist) * 50.0

        return penalty

    # ── Rewards ───────────────────────────────────────────────────────────
    def _reward_gk(self):
        gk   = self.our_team["gk"]
        ball = self.ball

        # Казна ако топката влезе во сопствениот гол
        if ball.x < GOAL_X_LEFT and GOAL_TOP < ball.y < GOAL_BOT:
            return -100.0

        reward = 0.0

        # Порамнување по Y со топката
        ideal_y = np.clip(ball.y, GK_Y_MIN, GK_Y_MAX)
        dy      = abs(gk.y - ideal_y)
        reward += max(0.0, 1.0 - dy / 120.0)

        #reward += self._proximity_penalty("gk")
        return reward

    # FIX 5: just_scored_goal параметар — само att и def добиваат гол reward
    def _reward_def(self, just_scored_goal=False):
        player = self.our_team["def"]
        ball   = self.ball
        reward = 0.0

        # FIX 5: def добива reward за гол (но не толку голем колку att)
        if just_scored_goal:
            return 50.0

        dist_to_ball = np.sqrt((ball.x - player.x)**2 + (ball.y - player.y)**2)

        # Трча кон топка
        # Наградувај ја компонентата на брзината кон десната врата
        ball_vx = self.ball.v * np.cos(self.ball.alpha)
        reward += np.clip(ball_vx / 200.0, -1.0, 1.0) * 3.0

#        # Допир во одбранбена зона
        touched = dist_to_ball < (player.radius + ball.radius + 5)
#        if touched and player.x < 683:
#            reward += 5.0

        # Топката оди напред по допир
        if touched and np.cos(ball.alpha) > 0.3 and ball.v > 50:
            reward += 3.0

        # Formation: стој помеѓу gk и att
#        gk  = self.our_team["gk"]
#        att = self.our_team["att"]
#        ideal_x = (gk.x + att.x) / 2.0
#        ideal_y = (gk.y + att.y) / 2.0
#        dist_ideal = np.sqrt((ideal_x - player.x)**2 + (ideal_y - player.y)**2)
#        reward += max(0.0, 0.5 - dist_ideal / 400.0)

        # Ако att и def се преклопуваат
        att = self.our_team["att"]
        min_att_def = player.radius + att.radius
        if self.curr_dist_att_def < min_att_def:
            if self.curr_dist_att_def > self.prev_dist_att_def:
                reward += 5.0
            else:
                reward -= 5.0

        reward += self._proximity_penalty("def")
        return reward

    # FIX 5: just_scored_goal параметар — att добива biggest reward за гол
    def _reward_att(self, just_scored_goal=False):
        player = self.our_team["att"]
        ball   = self.ball
        reward = 0.0

        # FIX 5+6: гол се регистрира преку just_scored_goal flag, не секој frame
        if just_scored_goal:
            return 100.0

        dist_to_ball = np.sqrt((ball.x - player.x)**2 + (ball.y - player.y)**2)

        # Допир со топката
        touched = dist_to_ball < (player.radius + ball.radius + 5)
        if touched:
            reward += 10.0
            if np.cos(ball.alpha) > 0.5 and ball.v > 30:
                reward += 3.0

        # Трча кон топката
        reward += max(0.0, 2.0 - dist_to_ball / 300.0)

        # Топката е блиску до противничката врата
        dist_ball_goal = np.sqrt((GOAL_X_RIGHT - ball.x)**2 + (GOAL_CY - ball.y)**2)
        reward += max(0.0, 2.0 - dist_ball_goal / 600.0)

        # блиску до спротивен гол -> продолжи напред, блиску до наш -> заобиколи
        dist_ball_our_goal   = np.sqrt((GOAL_X_LEFT  - ball.x)**2 + (GOAL_CY - ball.y)**2)
        dist_ball_their_goal = np.sqrt((GOAL_X_RIGHT - ball.x)**2 + (GOAL_CY - ball.y)**2)
        
        ball_vx = ball.v * np.cos(ball.alpha)  # позитивно = кон нивниот гол
        
        if dist_ball_our_goal < dist_ball_their_goal:
            # Топката е на наша страна — сакаме att да е ЛЕВО од топката
            # и да ја носи десно
            if player.x < ball.x:
                reward += 2.0  # добра позиција за да ја турка напред
            if ball_vx > 0:
                reward += ball_vx / 200.0 * 3.0  # топката оди кон нивниот гол
            if ball_vx < 0:
                reward -= abs(ball_vx) / 200.0 * 5.0  # казна — топката оди кон нашиот гол
        else:
            # Топката е на нивна страна — директен напад
            if touched:
                angle_to_goal = np.arctan2(GOAL_CY - ball.y, GOAL_X_RIGHT - ball.x)
                alignment     = np.cos(ball.alpha - angle_to_goal)
                reward       += alignment * 5.0
            if ball_vx > 0:
                reward += ball_vx / 200.0 * 4.0

        reward += self._proximity_penalty("att")
        return reward
