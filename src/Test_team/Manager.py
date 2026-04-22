import numpy as np

RIGHT_GOAL_X  = 1316
LEFT_GOAL_X   = 50
GOAL_TOP      = 343
GOAL_BOTTOM   = 578
GOAL_CENTER_Y = (GOAL_TOP + GOAL_BOTTOM) / 2
GK_MARGIN     = 40

def team_properties():
    properties = dict()
    properties['team_name'] = "Cutie.py"
    properties['player_names'] = ["Nate", "Maddie", "Cassie"]
    properties['image_name'] = 'Blue.png'
    properties['weight_points']           = (20, 15, 15)
    properties['radius_points']           = (20, 15, 20)
    properties['max_acceleration_points'] = (40, 10, 15)
    properties['max_speed_points']        = (40, 15, 25)
    properties['shot_power_points']       = (30, 25, 18)
    return properties

def goalkeeper_decision(player, ball, your_side):
    """Голманот се движи само горе-доле по линијата на голот."""
    gk_x = LEFT_GOAL_X + GK_MARGIN if your_side == 'left' else RIGHT_GOAL_X - GK_MARGIN

    target_y = np.clip(ball['y'], GOAL_TOP + player['radius'], GOAL_BOTTOM - player['radius'])

    dx = gk_x - player['x']
    dy = target_y - player['y']
    dist = np.sqrt(dx**2 + dy**2)

    return {
        'alpha': np.arctan2(dy, dx),
        'force': player['a_max'] * player['mass'] if dist > 5 else 0,
        'shot_request': False,
        'shot_power': 0
    }

def decision(our_team, their_team, ball, your_side, half, time_left, our_score, their_score):
    attack_goal_x = RIGHT_GOAL_X if your_side == 'left' else LEFT_GOAL_X
    defend_goal_x = LEFT_GOAL_X  if your_side == 'left' else RIGHT_GOAL_X

    manager_decision = [dict(), dict(), dict()]

    for i in range(3):
        player = our_team[i]

        # --- Голман ---
        if i == 0:
            manager_decision[i] = goalkeeper_decision(player, ball, your_side)
            continue

        dist_to_ball = np.sqrt((ball['x'] - player['x'])**2 + (ball['y'] - player['y'])**2)
        alpha_to_ball = np.arctan2(ball['y'] - player['y'], ball['x'] - player['x'])
        alpha_to_goal = np.arctan2(GOAL_CENTER_Y - player['y'], attack_goal_x - player['x'])
        near_ball = dist_to_ball < 80

        # --- Дефендер (i=1): останува во сопствената половина ---
        if i == 1:
            mid_x = (LEFT_GOAL_X + RIGHT_GOAL_X) / 2

            # Целна позиција: помеѓу голот и средината, следи Y на топката
            defend_x = defend_goal_x + (GK_MARGIN + 150) * (1 if your_side == 'left' else -1)
            defend_y = np.clip(ball['y'], GOAL_TOP, GOAL_BOTTOM)

            dx = defend_x - player['x']
            dy = defend_y - player['y']
            dist_to_pos = np.sqrt(dx**2 + dy**2)

            # Ако топката е во сопствената половина – оди кон неа
            ball_in_own_half = (ball['x'] < mid_x) if your_side == 'left' else (ball['x'] > mid_x)

            if ball_in_own_half and near_ball:
                manager_decision[i] = {
                    'alpha': alpha_to_goal,  # удри кон противничкиот гол
                    'force': player['a_max'] * player['mass'],
                    'shot_request': True,
                    'shot_power': player['shot_power_max']
                }
            elif ball_in_own_half:
                manager_decision[i] = {
                    'alpha': alpha_to_ball,
                    'force': player['a_max'] * player['mass'],
                    'shot_request': False,
                    'shot_power': 0
                }
            else:
                # Топката е во противничката половина – остани на позиција
                manager_decision[i] = {
                    'alpha': np.arctan2(dy, dx),
                    'force': player['a_max'] * player['mass'] if dist_to_pos > 10 else 0,
                    'shot_request': False,
                    'shot_power': 0
                }
            continue

        # --- Напаѓач (i=2): секогаш притискај ---
        manager_decision[i] = {
            'alpha': alpha_to_goal if near_ball else alpha_to_ball,
            'force': player['a_max'] * player['mass'],
            'shot_request': near_ball,
            'shot_power': player['shot_power_max']
        }

    return manager_decision
