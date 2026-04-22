# eval.py
# Тестира колку голови прави тимот во N headless игри.
# Стартувај со: python eval.py

import os
import sys
import numpy as np
from stable_baselines3 import PPO

from football_env import AIFootballEnv

# ── Конфигурација ──────────────────────────────────────────────────────────
N_GAMES   = 10     # Број на игри за евалуација
MAX_STEPS = 3000   # Чекори по епизода (исто како во тренинг)
MODEL_DIR = "models"

# ── Вчитај модели ──────────────────────────────────────────────────────────
models = {}
for ag in ["gk", "def", "att"]:
    path = os.path.join(MODEL_DIR, f"ppo_{ag}.zip")
    if os.path.exists(path):
        models[ag] = PPO.load(path)
        print(f"✅ Вчитан: {path}")
    else:
        models[ag] = None
        print(f"⚠️  Нема модел: {path} — ќе се користи случајна акција")

# ── Евалуација ─────────────────────────────────────────────────────────────
def run_eval():
    results = {
        "goals_scored":    0,
        "goals_conceded":  0,
        "episodes_with_goal": 0,
        "total_steps":     0,
        "total_rewards":   {"gk": 0.0, "def": 0.0, "att": 0.0},
    }

    print(f"\n{'='*50}")
    print(f"  Евалуација: {N_GAMES} игри")
    print(f"{'='*50}\n")

    env = AIFootballEnv()

    for game_i in range(N_GAMES):
        obs_dict, _ = env.reset()
        ep_rewards  = {"gk": 0.0, "def": 0.0, "att": 0.0}
        steps       = 0
        scored      = False

        while env.agents:
            actions = {}
            for ag in env.agents:
                model = models.get(ag)
                if model:
                    act, _ = model.predict(obs_dict[ag], deterministic=True)
                else:
                    act = env.action_space(ag).sample()
                actions[ag] = act

            obs_dict, rewards, terminations, truncations, infos = env.step(actions)

            for ag, r in rewards.items():
                ep_rewards[ag] += r

            steps += 1

            # Провери дали е постигнат гол
            if env.goal_scored and not scored:
                results["goals_scored"]       += 1
                results["episodes_with_goal"] += 1
                scored = True

        results["total_steps"] += steps
        for ag in ["gk", "def", "att"]:
            results["total_rewards"][ag] += ep_rewards[ag]

        # Печати резултат по игра
        goal_str = "⚽ ГОЛ!" if scored else "——"
        print(f"  Игра {game_i+1:>2}/{N_GAMES}  |  Чекори: {steps:>4}  |  "
              f"Награди → gk:{ep_rewards['gk']:>7.1f}  "
              f"def:{ep_rewards['def']:>7.1f}  "
              f"att:{ep_rewards['att']:>7.1f}  |  {goal_str}")

    # ── Финален извештај ───────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  ФИНАЛЕН ИЗВЕШТАЈ")
    print(f"{'='*50}")
    print(f"  Вкупно игри:          {N_GAMES}")
    print(f"  Игри со гол:          {results['episodes_with_goal']} / {N_GAMES}  "
          f"({100*results['episodes_with_goal']/N_GAMES:.0f}%)")
    print(f"  Просечни чекори:      {results['total_steps']/N_GAMES:.0f} / {MAX_STEPS}")
    print(f"\n  Просечна награда по игра:")
    for ag in ["gk", "def", "att"]:
        avg = results["total_rewards"][ag] / N_GAMES
        print(f"    {ag:>4}: {avg:>8.2f}")
    print(f"{'='*50}\n")

    return results


if __name__ == "__main__":
    run_eval()
