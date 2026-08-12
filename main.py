import gymnasium as gym, numpy as np
import numpy as np
import matplotlib.pyplot as plt
import torch, torch.nn as nn

plt.rcParams.update({"font.size": 14, "figure.dpi": 110})

env = gym.make("CartPole-v1")
obs, _ = env.reset()
# print(obs)
# print(obs.dtype, env.observation_space.low, env.observation_space.high)

net = nn.Sequential(nn.Linear(4,64), nn.ReLU(), nn.Linear(64,2))

n = 1
sum = 0
vec = []
for i in range(n):
    obs, _ = env.reset()
    done = False
    cnt = 0
    while not done:
        q_val = net(
            torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            ) # 神秘batch
        act = q_val.argmax(dim=1).item()
        next_obs, reward, terminated, truncated, _ = env.step(act)
        obs = next_obs
        done = terminated or truncated
        cnt += 1
    sum += cnt
    vec.append(cnt)

fig, axes = plt.subplots(1, 1, figsize=(10, 7), sharey=True, sharex=True)
# ax = axes[0]
axes.plot([sum/n for _ in range(n)])
axes.plot(vec)
# fig.savefig('./figs/random.png', dpi=150, bbox_inches='tight')
plt.show()
