import numpy as np

RIGHT_GOAL_X  = 1316
LEFT_GOAL_X   = 50
GOAL_TOP      = 343
GOAL_BOTTOM   = 578
GOAL_CENTER_Y = (GOAL_TOP + GOAL_BOTTOM) / 2

def team_properties():
    properties = dict()
    player_names = ["Пандевалдо", "Панчевалдо", "Елмасалдо"]
    properties['team_name'] = "Мак Челзи"
    properties['player_names'] = player_names
    properties['image_name'] = 'Red.png'
    properties['weight_points']           = (3,  25, 15) #prv e golmanot, vtor e defender, tret napagjach
    properties['radius_points']           = (35,  25, 5)
    properties['max_acceleration_points'] = (3, 7, 25)
    properties['max_speed_points']        = (35, 7, 25)
    properties['shot_power_points']       = (20, 15, 25)
    return properties

def decision(our_team, their_team, ball, your_side, half, time_left, our_score, their_score):
   
    attack_goal_x = RIGHT_GOAL_X if your_side == 'left' else LEFT_GOAL_X

    manager_decision = [dict(), dict(), dict()]

    for i in range(3):
        player = our_team[i]

        # Растојание до топката
        dist_to_ball = np.sqrt(
            (ball['x'] - player['x'])**2 +
            (ball['y'] - player['y'])**2
        )

        # Агол: играч → топка
        alpha_to_ball = np.arctan2(
            ball['y'] - player['y'],
            ball['x'] - player['x']
        )

        # Агол: топка → врата на противникот
        alpha_to_goal = np.arctan2(
            GOAL_CENTER_Y - player['y'],
            attack_goal_x  - player['x']
        )

        near_ball = dist_to_ball < 80

        manager_decision[i]['alpha']        = alpha_to_goal if near_ball else alpha_to_ball
        manager_decision[i]['force']        = player['a_max'] * player['mass']
        manager_decision[i]['shot_request'] = near_ball
        manager_decision[i]['shot_power']   = player['shot_power_max']

    return manager_decision