export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.7 && \
python ../../train_rlpd.py "$@" \
    --exp_name=tennis_ball_pick \
    --checkpoint_path=/home/ruiqiang/workspaces/HK_TacExo_HAN/hil-serl/examples/ckpt-pick-1107/2025-11-06_online_rlpd \
    --demo_path=../../demo_data/tennis_ball_pick_19_demos_2025-11-03_20-04-09.pkl \
    --learner \