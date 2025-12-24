import pickle as pkl, numpy as np
p = "/home/ruiqiang/workspaces/HK_TacExo_HAN/hil-serl/examples/demo_data/tennis_ball_pick_18_demos_2025-10-16_18-44-43_fixed.pkl"
with open(p, "rb") as f:
    traj = pkl.load(f)
print("num transitions:", len(traj))
print("sample keys:", traj[0].keys())
obs = traj[0]["observations"]
print("obs keys:", list(obs.keys()))
for k in ["front_camera","tactile_data","gaze_mask","state"]:
    v = obs.get(k, None)
    print(k, type(v), None if v is None else (np.array(v).shape, np.array(v).dtype))
