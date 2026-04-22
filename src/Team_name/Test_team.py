# Test_team.py
# Го користи тренираниот PPO модел за секој агент (gk, def, att).
# Ги вчитува моделите од папката models/ и ги повикува во decision().

import os
import numpy as np
from stable_baselines3 import PPO

# ── Константи (исти како во football_env.py) ──────────────────────────────
FIELD_W      = 1366.0
FIELD_H      = 768.0
GOAL_X_LEFT  = 50
GOAL_X_RIGHT = 1316
GOAL_TOP     = 343
GOAL_BOT     = 578
GOAL_CY      = (GOAL_TOP + GOAL_BOT) / 2
GK_X_LOCK    = 85
GK_Y_MIN     = GOAL_TOP
GK_Y_MAX     = GOAL_BOT

# ── Вчитај модели еднаш при импорт ────────────────────────────────────────
_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

_models = {}
for _ag in ["gk", "def", "att"]:
    _path = os.path.join(_base, f"ppo_{_ag}.zip")
    if os.path.exists(_path):
        _models[_ag] = PPO.load(_path)
        print(f"[Test_team] Вчитан модел: {_path}")
    else:
        _models[_ag] = None
        print(f"[Test_team] ВНИМАНИЕ: Нема модел на {_path}, ќе се користи fallback логика")


# ── Team properties ────────────────────────────────────────────────────────
def team_properties():
    properties = dict()
    properties['team_name']    = "Cutie.py"
    properties['player_names'] = ["Nate", "Maddie", "Cassie"]
    properties['image_name']   = 'Blue.png'
    properties['weight_points']           = (20, 15, 15)
    properties['radius_points']           = (20, 15, 20)
    properties['max_acceleration_points'] = (40, 10, 15)
    properties['max_speed_points']        = (40, 15, 25)
    properties['shot_power_points']       = (30, 25, 18)
    return properties


# ── Observation builders (идентични со football_env.py) ───────────────────
def _nx(v): return (v / FIELD_W) * 2.0 - 1.0
def _ny(v): return (v / FIELD_H) * 2.0 - 1.0
def _nv(v, mx): return float(np.clip(v / max(mx, 1), 0, 1)) * 2.0 - 1.0


def _obs_gk(player, ball):
    dist = np.sqrt((ball['x'] - player['x'])**2 + (ball['y'] - player['y'])**2)
    return np.array([
        _ny(player['y']),
        _nx(ball['x']),
        _ny(ball['y']),
        _nv(ball['v'], 850),
        float(np.clip(dist / FIELD_W, 0, 1)) * 2 - 1,
    ], dtype=np.float32)


def _obs_def(player, ball, other):
    """other = att player dict"""
    return np.array([
        _nx(player['x']),    _ny(player['y']),
        _nv(player['v'], player['v_max']),
        player['alpha'] / np.pi,
        _nx(ball['x']),      _ny(ball['y']),
        _nv(ball['v'], 850),
        ball['alpha'] / np.pi,
        _nx(other['x']),     _ny(other['y']),
        _nx(GOAL_X_RIGHT),   _ny(GOAL_CY),
    ], dtype=np.float32)


def _obs_att(player, ball, other):
    """other = def player dict"""
    return np.array([
        _nx(player['x']),    _ny(player['y']),
        _nv(player['v'], player['v_max']),
        player['alpha'] / np.pi,
        _nx(ball['x']),      _ny(ball['y']),
        _nv(ball['v'], 850),
        ball['alpha'] / np.pi,
        _nx(other['x']),     _ny(other['y']),
        _nx(GOAL_X_RIGHT),   _ny(GOAL_CY),
    ], dtype=np.float32)


# ── Fallback логика (ако модел не постои) ─────────────────────────────────
def _fallback_gk(player, ball, your_side):
    gk_x   = GK_X_LOCK if your_side == 'left' else FIELD_W - GK_X_LOCK
    target_y = float(np.clip(ball['y'], GK_Y_MIN, GK_Y_MAX))
    dx     = gk_x   - player['x']
    dy     = target_y - player['y']
    dist   = np.sqrt(dx**2 + dy**2)
    return {
        'alpha':        np.arctan2(dy, dx),
        'force':        player['a_max'] * player['mass'] if dist > 5 else 0,
        'shot_request': False,
        'shot_power':   0,
    }


def _fallback_field(player, ball, attack_goal_x):
    dist  = np.sqrt((ball['x'] - player['x'])**2 + (ball['y'] - player['y'])**2)
    near  = dist < 80
    alpha = np.arctan2(GOAL_CY - player['y'], attack_goal_x - player['x']) if near \
            else np.arctan2(ball['y'] - player['y'], ball['x'] - player['x'])
    return {
        'alpha':        alpha,
        'force':        player['a_max'] * player['mass'],
        'shot_request': near,
        'shot_power':   player['shot_power_max'],
    }


# ── Action → decision конвертор ────────────────────────────────────────────
def _action_to_decision(action, player, agent_id, your_side):
    max_force      = 200 * 0.75
    max_shot_power = 200 * 0.95

    if agent_id == "gk":
        # GK акција: [y_direction]
        # Го претвораме во движење горе/доле
        gk_x    = GK_X_LOCK if your_side == 'left' else FIELD_W - GK_X_LOCK
        target_y = float(np.clip(
            player['y'] + action[0] * player['v_max'] * (1/60),
            GK_Y_MIN, GK_Y_MAX
        ))
        dx = gk_x       - player['x']
        dy = target_y   - player['y']
        return {
            'alpha':        np.arctan2(dy, dx),
            'force':        player['a_max'] * player['mass'],
            'shot_request': False,
            'shot_power':   0,
        }
    else:
        # DEF / ATT акција: [force, alpha, shot_power, shot_request]
        return {
            'force':        float(action[0]) * max_force,
            'alpha':        float(action[1]) * np.pi,
            'shot_power':   abs(float(action[2])) * max_shot_power,
            'shot_request': float(action[3]) > 0,
        }


# ── Главна decision функција ───────────────────────────────────────────────
def decision(our_team, their_team, ball, your_side, half, time_left, our_score, their_score):
    """
    our_team[0] = gk  (Nate)
    our_team[1] = def (Maddie)
    our_team[2] = att (Cassie)
    """
    manager_decision = [dict(), dict(), dict()]

    agent_ids = ["gk", "def", "att"]

    for i, agent_id in enumerate(agent_ids):
        player = our_team[i]
        model  = _models.get(agent_id)

        if model is None:
            # Fallback ако моделот не е вчитан
            if agent_id == "gk":
                manager_decision[i] = _fallback_gk(player, ball, your_side)
            else:
                attack_goal_x = GOAL_X_RIGHT if your_side == 'left' else GOAL_X_LEFT
                manager_decision[i] = _fallback_field(player, ball, attack_goal_x)
            continue

        # Изгради опсервација
        if agent_id == "gk":
            obs = _obs_gk(player, ball)

        elif agent_id == "def":
            att_player = our_team[2]   # att е индекс 2
            obs = _obs_def(player, ball, att_player)

        else:  # att
            def_player = our_team[1]   # def е индекс 1
            obs = _obs_att(player, ball, def_player)

        # Ако тимот игра на десна страна — страните се свапувани во half 2
        # Ги инвертираме x-координатите за да добиеме конзистентна перспектива
        if your_side == 'right':
            obs = _mirror_obs(obs, agent_id)

        # PPO предвидување
        action, _ = model.predict(obs, deterministic=True)

        # Конвертирај акција во decision речник
        dec = _action_to_decision(action, player, agent_id, your_side)

        # Ако тимот е на десна страна, инвертирај ги насоките
        if your_side == 'right':
            dec['alpha'] = np.pi - dec['alpha']

        manager_decision[i] = dec
        
    return manager_decision


# ── Миророрирање на опсервација за десна страна ────────────────────────────
def _mirror_obs(obs, agent_id):
    """Инвертира ги x-вредностите во опсервацијата кога тимот е на десна страна."""
    obs = obs.copy()
    if agent_id == "gk":
        # [player_y, ball_x, ball_y, ball_v, dist]
        obs[1] = -obs[1]   # ball_x
    else:
        # [px, py, pv, palpha, bx, by, bv, balpha, ox, oy, goal_x, goal_y]
        obs[0]  = -obs[0]   # player x
        obs[3]  = -obs[3]   # player alpha
        obs[4]  = -obs[4]   # ball x
        obs[7]  = -obs[7]   # ball alpha
        obs[8]  = -obs[8]   # other player x
        obs[10] = -obs[10]  # goal x
    return obs
