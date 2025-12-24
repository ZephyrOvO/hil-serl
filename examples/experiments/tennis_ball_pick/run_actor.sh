export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
python ../../train_rlpd.py "$@" \
    --exp_name=tennis_ball_pick \
    --checkpoint_path=/home/ruiqiang/workspaces/HK_TacExo_HAN/hil-serl/examples/ckpt-pick-1107/2025-11-06_online_rlpd \
    --actor \
