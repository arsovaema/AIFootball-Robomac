# train.py
# Independent PPO — секој агент (gk, def, att) си има своја PPO политика.
# Тренирање: ~5-10 минути на обичен CPU.
# Потребни пакети: pip install stable-baselines3 pettingzoo supersuit gymnasium

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from football_env import AIFootballEnv

# ── Конфигурација ──────────────────────────────────────────────────────────
TOTAL_TIMESTEPS = 200_000   # ~5-10 мин на CPU; зголеми на 500k+ за подобри резултати
SAVE_DIR        = "models"
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Single-agent wrapper ───────────────────────────────────────────────────
# Го изолира еден агент во стандарден Gymnasium интерфејс.
# Другите два агенти користат случајни акции за време на тренингот.
# Редоследот на тренирање: прво att, потоа def, потоа gk.

class SingleAgentWrapper(gym.Env):
    """
    Извлекува еден агент од AIFootballEnv и го изложува
    како стандарден Gymnasium Env. Другите агенти дејствуваат
    со последно научената политика (или случајно ако уште нема).
    """

    def __init__(self, agent_id: str, other_policies: dict):
        super().__init__()
        self.agent_id       = agent_id
        self.other_policies = other_policies   # {"gk": PPO, "def": PPO, ...} или None

        self._env = AIFootballEnv()
        self.observation_space = self._env.observation_space(agent_id)
        self.action_space      = self._env.action_space(agent_id)

    def reset(self, seed=None, options=None):
        self._obs_dict, infos = self._env.reset(seed=seed)
        return self._obs_dict[self.agent_id], {}

    def step(self, action):
        # Составување акции за сите агенти
        actions = {}
        for ag in self._env.possible_agents:
            if ag == self.agent_id:
                actions[ag] = action
            else:
                policy = self.other_policies.get(ag)
                if policy is not None:
                    obs        = self._obs_dict[ag]
                    act, _     = policy.predict(obs, deterministic=True)
                    actions[ag] = act
                else:
                    # Случајна акција ако политиката уште не постои
                    actions[ag] = self._env.action_space(ag).sample()

        self._obs_dict, rewards, terminations, truncations, infos = self._env.step(actions)

        reward   = rewards.get(self.agent_id, 0.0)
        done     = terminations.get(self.agent_id, False)
        truncated = truncations.get(self.agent_id, False)

        obs = self._obs_dict.get(self.agent_id,
              self._env.observation_space(self.agent_id).sample() * 0.0)

        return obs, reward, done, truncated, infos.get(self.agent_id, {})

    def render(self): pass
    def close(self):  self._env.close() if hasattr(self._env, "close") else None


# ── Помошна функција за правење VecEnv ────────────────────────────────────
def make_vec_env(agent_id, other_policies, n_envs=4):
    def make():
        return SingleAgentWrapper(agent_id, other_policies)
    envs = DummyVecEnv([make] * n_envs)
    return VecMonitor(envs)


# ── PPO хиперпараметри (брзо тренирање) ───────────────────────────────────
PPO_KWARGS = dict(
    policy          = "MlpPolicy",
    learning_rate   = 3e-4,
    n_steps         = 512,      # помал buffer → почест update → побрзо
    batch_size      = 128,
    n_epochs        = 5,
    gamma           = 0.99,
    gae_lambda      = 0.95,
    clip_range      = 0.2,
    ent_coef        = 0.01,     # малку ентропија за истражување
    verbose         = 1,
    policy_kwargs   = dict(net_arch=[64, 64]),  # мала мрежа → побрзо
)


# ── Главен тренинг ─────────────────────────────────────────────────────────
def train():
    policies = {}   # ги чува научените политики

    # Редослед: att прв — тој има најјасна награда (гол = +100)
    # потоа def, потоа gk
    training_order = [
        ("att", TOTAL_TIMESTEPS),
        ("def", TOTAL_TIMESTEPS),
        ("gk",  TOTAL_TIMESTEPS),
    ]

    for agent_id, timesteps in training_order:
        print(f"\n{'='*50}")
        print(f"  Тренирање: {agent_id.upper()}  ({timesteps:,} чекори)")
        print(f"{'='*50}")

        # Другите политики ги земаме од веќе научените (или None)
        other_policies = {k: v for k, v in policies.items() if k != agent_id}

        vec_env = make_vec_env(agent_id, other_policies, n_envs=4)

        model = PPO(env=vec_env, **PPO_KWARGS)
        model.learn(total_timesteps=timesteps, progress_bar=True)

        save_path = os.path.join(SAVE_DIR, f"ppo_{agent_id}")
        model.save(save_path)
        print(f"  Зачувано: {save_path}.zip")

        policies[agent_id] = model
        vec_env.close()

    print("\n✅ Тренирањето е завршено! Моделите се во папката 'models/'")
    return policies


# ── Eval: играј 1 епизода со сите научени политики ────────────────────────
def evaluate(policies):
    print("\n── Евалуација ──")
    env  = AIFootballEnv()
    obs_dict, _ = env.reset()
    total_rewards = {ag: 0.0 for ag in env.possible_agents}
    step_count = 0

    while env.agents:
        actions = {}
        for ag in env.agents:
            policy = policies.get(ag)
            if policy:
                act, _ = policy.predict(obs_dict[ag], deterministic=True)
            else:
                act = env.action_space(ag).sample()
            actions[ag] = act

        obs_dict, rewards, terminations, truncations, _ = env.step(actions)
        for ag, r in rewards.items():
            total_rewards[ag] += r
        step_count += 1

    print(f"  Чекори: {step_count}")
    for ag, total in total_rewards.items():
        print(f"  {ag:>4}: вкупна награда = {total:.2f}")


# ── Влезна точка ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    trained_policies = train()
    evaluate(trained_policies)