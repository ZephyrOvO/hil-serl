import glob
import os, sys
import pickle as pkl
import jax
from jax import numpy as jnp
import flax.linen as nn
from flax.training import checkpoints
import numpy as np
import optax
from tqdm import tqdm
from absl import app, flags
import gymnasium as gym

from serl_launcher.data.data_store import ReplayBuffer
from serl_launcher.utils.train_utils import concat_batches
from serl_launcher.vision.data_augmentations import batched_random_crop
from serl_launcher.networks.reward_classifier_gaze import create_classifier as create_classifier_reward

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../serl_robot_infra'))
sys.path.insert(0, project_root)
from experiments.mappings import NEW_MAPPING

# NEW: 我们的扩展版（含 task="gaze"）
from serl_launcher.networks.reward_classifier_gaze import create_classifier as create_classifier_extended

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", "tennis_ball_pick", "Name of experiment corresponding to folder.")
flags.DEFINE_integer("num_epochs", 50, "Number of training epochs.")
flags.DEFINE_integer("ckpt_every", 100, "Save checkpoint every N epochs; 0 = only final")
flags.DEFINE_integer("batch_size", 256, "Batch size.")
flags.DEFINE_integer("is_pick_task", 0, "evaluate pick or place task.")
flags.DEFINE_integer("is_pick_and_place_task", 1, "evaluate pick or place task.")
flags.DEFINE_string("gaze_data_root_train", "./gaze_cls_data/train", "gaze训练集目录（pkl）")
flags.DEFINE_string("gaze_data_root_val", "./gaze_cls_data/val", "gaze验证集目录（pkl）")

# NEW: gaze 训练相关
flags.DEFINE_string("train_task", "reward", "reward | gaze")
flags.DEFINE_string("gaze_data_root", "./gaze_cls_data", "export_gaze_dataset.py 生成的根目录")
flags.DEFINE_float("cls_loss_weight", 1.0, "三分类交叉熵的权重")
flags.DEFINE_float("box_loss_weight", 1.0, "bbox回归 L1 的权重（仅对 hit!=none 样本）")
flags.DEFINE_float("none_drop_prob", 0.0, "采样时丢弃 none 类(2)的概率")
flags.DEFINE_integer("steps_per_epoch", 200, "每个 epoch 训练的 batch 数（gaze 任务）")
flags.DEFINE_boolean("use_class_weight", False, "是否对三分类使用类别权重")
flags.DEFINE_string("ckpt_root", "/mnt/data3/hcj",
                    "gaze 训练 ckpt 的根目录（会在里面新建 gaze_ckpt 子目录）")

def _iter_gaze_shards(root: str):
    for p in sorted(glob.glob(os.path.join(root, "*.pkl"))):
        with open(p, "rb") as f:
            shard = pkl.load(f)   # list[dict]
        yield shard

def _gaze_batch_iterator(root: str, batch_size: int, none_drop_prob: float = 0.0, image_key: str = "front_camera"):
    """
    读取 export_gaze_dataset.py 生成的分片，打乱并做简单的按类均衡采样。
    目标比例默认 1:1:1（可按需要调整），none_drop_prob 仍然生效（降低 none 出现几率）。
    """
    rng = np.random.RandomState(0)

    # 三个类别的桶
    buckets = {0: [], 1: [], 2: []}

    # 目标比例（可以按需改，比如把 none 降低到 0.5 份）
    target_ratio = {0: 1, 1: 1, 2: 1}
    ratio_sum = sum(target_ratio.values())

    def maybe_yield_batch():
        # 根据目标比例，从各桶取样组成一个 batch
        nonlocal buckets
        need = {c: max(1, batch_size * target_ratio[c] // ratio_sum) for c in buckets}
        # 补上整数除法可能的欠缺
        total_need = sum(need.values())
        if total_need < batch_size:
            # 把缺的补到非 none 类（优先 0、1）
            for c in (0, 1, 2):
                while total_need < batch_size:
                    need[c] += 1
                    total_need += 1

        ok = all(len(buckets[c]) >= need[c] for c in buckets)
        if not ok:
            return None

        imgs, y_cls, y_box = [], [], []
        for c in (0, 1, 2):
            take_idx = rng.choice(len(buckets[c]), size=need[c], replace=False)
            take_idx = np.sort(take_idx)[::-1]  # 逆序删除
            for idx in take_idx:
                ex = buckets[c].pop(idx)
                imgs.append(ex["img"])
                y_cls.append(ex["hit"])
                y_box.append(ex["box"])

        imgs = (np.asarray(imgs, np.float32) / 255.0)
        y_cls = np.asarray(y_cls, np.int32)
        y_box = np.asarray(y_box, np.float32)         # (B,4) in [0,1] or (-1,-1,-1,-1)
        return {"front_camera": imgs}, {"hit": y_cls, "box": y_box}

    # 遍历文件，逐 shard 读入并打乱
    for pkl_path in sorted(glob.glob(os.path.join(root, "*.pkl"))):
        with open(pkl_path, "rb") as f:
            shard = pkl.load(f)

        # 打乱该 shard
        idx = np.arange(len(shard))
        rng.shuffle(idx)

        for i in idx:
            ex = shard[i]
            c = int(ex["gaze_hit_class"])
            if c == 2 and rng.rand() < none_drop_prob:
                continue
            buckets[c].append({
                "img": ex["observations"][image_key],
                "hit": c,
                "box": np.asarray(ex["gaze_box"], np.float32),
            })

            out = maybe_yield_batch()
            if out is not None:
                yield out

    # 把剩余的样本尽量再拼几个 batch（可选）
    while True:
        out = maybe_yield_batch()
        if out is None:
            break
        yield out
        
        
def _gaze_batch_iterator_sequential(root: str, batch_size: int, image_key: str = "front_camera"):
    """顺序读取 shards，不做类均衡。用于验证/测试以保持真实分布。"""
    buf_img, buf_cls, buf_box = [], [], []
    for pkl_path in sorted(glob.glob(os.path.join(root, "*.pkl"))):
        with open(pkl_path, "rb") as f:
            shard = pkl.load(f)
        for ex in shard:
            buf_img.append(ex["observations"][image_key])
            buf_cls.append(int(ex["gaze_hit_class"]))
            buf_box.append(np.asarray(ex["gaze_box"], np.float32))
            if len(buf_img) == batch_size:
                imgs = (np.asarray(buf_img, np.float32) / 255.0)
                y_cls = np.asarray(buf_cls, np.int32)
                y_box = np.asarray(buf_box, np.float32)
                yield {"front_camera": imgs}, {"hit": y_cls, "box": y_box}
                buf_img, buf_cls, buf_box = [], [], []
    if buf_img:
        imgs = (np.asarray(buf_img, np.float32) / 255.0)
        y_cls = np.asarray(buf_cls, np.int32)
        y_box = np.asarray(buf_box, np.float32)
        yield {"front_camera": imgs}, {"hit": y_cls, "box": y_box}



def eval_epoch_gaze_box(state, root: str, batch_size: int = 256, image_key: str = "front_camera"):
    """
    遍历整个 root（val pkl 目录），返回：
      acc_mean：整体验证分类准确率
      mae_box  ：仅对 hit<2（obj0/obj1）的样本计算的 |x-x̂|+|y-ŷ| 的平均
    """
    it = _gaze_batch_iterator(root, batch_size, none_drop_prob=0.0, image_key=image_key)
    acc_sum, mae_sum, n, valid_mae_cnt = 0.0, 0.0, 0, 0
    for imgs, labels in it:
        batch = {
            "front_camera": imgs["front_camera"].astype(np.float32),
            "hit": labels["hit"].astype(np.int32),
            "box":  labels["box"].astype(np.float32),
        }
        logits, box_eval = state.apply_fn({"params": state.params},
                                         {"front_camera": batch["front_camera"]},
                                         train=False)
        # to numpy
        logits = np.array(logits); box_eval = np.array(box_eval)
        y_cls = np.array(batch["hit"]); y_box = np.array(batch["box"])
        # 分类 acc
        acc = (np.argmax(logits, axis=-1) == y_cls).astype(np.float32).mean()
        acc_sum += float(acc); n += 1
        # bbox MAE（L1：4 个坐标之和；只对 obj0/obj1）
        m = (y_cls < 2).astype(np.float32)
        tot = m.sum()
        if tot > 0:
            mae = (np.abs(box_eval - y_box).sum(axis=-1) * m).sum() / tot
            mae_sum += float(mae); valid_mae_cnt += 1
    acc_mean = acc_sum / max(n, 1)
    mae_mean = mae_sum / max(valid_mae_cnt, 1) if valid_mae_cnt > 0 else float("nan")
    return acc_mean, mae_mean


def main(_):
    assert FLAGS.exp_name in NEW_MAPPING, 'Experiment folder not found.'
    config = NEW_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=True, save_video=False, classifier=False)

    devices = jax.local_devices()
    sharding = jax.sharding.PositionalSharding(devices)

    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)

    # 用环境 sample 推断网络输入 shape
    sample_obs_space = env.observation_space.sample()

    if FLAGS.train_task == "gaze":
        # -------- 构建 gaze 模型 --------
        state = create_classifier_extended(
            key,
            {"front_camera": np.zeros((1, 128, 128, 3), np.float32)},
            image_keys=["front_camera"],
            image_key_weights=None,
            task="gaze_box",
        )

        # -------- 定义 gaze 的 train_step --------
        def _box_mask(y_cls: jnp.ndarray) -> jnp.ndarray:
            # 只对 obj0/obj1 (0/1) 做 bbox 回归；none(2) 屏蔽
            return (y_cls < 2).astype(jnp.float32)[..., None]   # (B,1)
        @jax.jit
        def train_step_gaze_box(state, batch, key, cls_w: float, box_w: float, class_weight: jnp.ndarray | None):
            y_cls = batch["hit"]        # (B,)
            y_box = batch["box"]        # (B,4)
            m_box = _box_mask(y_cls)    # (B,1)

            def loss_fn(params):
                logits, box_pred = state.apply_fn({"params": params},
                                                {"front_camera": batch["front_camera"]},
                                                rngs={"dropout": key},
                                                train=True)
                ce_per  = optax.softmax_cross_entropy_with_integer_labels(logits, y_cls)
                if class_weight is not None:
                    w = class_weight[y_cls]
                    ce = (ce_per * w).mean()
                else:
                    ce = ce_per.mean()
                l1  = jnp.abs(box_pred - y_box).sum(axis=-1, keepdims=True)  # (B,1)
                reg = (l1 * m_box).sum(axis=-1) / (m_box.sum(axis=-1) + 1e-8) # (B,)
                reg = reg.mean()
                return cls_w * ce + box_w * reg, (ce, reg)

            (total, (ce, reg)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            # 不再 freeze，直接用 dict 结构的 grads
            new_state = state.apply_gradients(grads=grads)

            logits_eval, box_eval = new_state.apply_fn({"params": new_state.params},
                                                    {"front_camera": batch["front_camera"]},
                                                    rngs={"dropout": key},
                                                    train=False)
            acc = jnp.mean((jnp.argmax(logits_eval, axis=-1) == y_cls).astype(jnp.float32))
            reg_mask = m_box.squeeze(-1)
            reg_l1 = (jnp.abs(box_eval - y_box).sum(axis=-1) * reg_mask).sum() / (reg_mask.sum() + 1e-8)
            return new_state, total, ce, reg, acc, reg_l1



        # 统计两条“最佳曲线”的记录
        best_train = {"acc": -1.0, "epoch": -1, "val_at_best": float("nan")}
        best_val   = {"acc": -1.0, "epoch": -1, "train_at_best": float("nan")}
        # class weight（可选）：轻度提高 none(2) 的权重以抑制误报
        class_weight = None
        if FLAGS.use_class_weight:
            class_weight = jnp.array([1.0, 1.0, 1.5], dtype=jnp.float32)
        # it = _gaze_batch_iterator(FLAGS.gaze_data_root, FLAGS.batch_size, FLAGS.none_drop_prob, image_key="front_camera")
        it = _gaze_batch_iterator(FLAGS.gaze_data_root_train, FLAGS.batch_size, FLAGS.none_drop_prob, image_key="front_camera")
        for epoch in tqdm(range(FLAGS.num_epochs)):
            step = 0
            # === 训练集 epoch 累计器 ===
            epoch_acc_sum = 0.0
            epoch_regl1_sum = 0.0
            epoch_cnt = 0
            while step < FLAGS.steps_per_epoch:
                try:
                    imgs, labels = next(it)   # imgs: {"front_camera": (B,H,W,3)}, labels: {"hit":(B,), "box":(B,4)}
                except StopIteration:
                    # it = _gaze_batch_iterator(FLAGS.gaze_data_root, FLAGS.batch_size, FLAGS.none_drop_prob, image_key="front_camera")
                    # imgs, labels = next(it)
                    it = _gaze_batch_iterator(FLAGS.gaze_data_root_train, FLAGS.batch_size, FLAGS.none_drop_prob, image_key="front_camera")
                    imgs, labels = next(it)
                
                batch = {
                    "front_camera": imgs["front_camera"].astype(np.float32),
                    "hit": labels["hit"].astype(np.int32),
                    "box": labels["box"].astype(np.float32),
                }
                rng, key = jax.random.split(rng)
                state, total, ce, reg, acc, reg_l1 = train_step_gaze_box(
                    state, batch, key, FLAGS.cls_loss_weight, FLAGS.box_loss_weight, class_weight
                )
                if (step % 10) == 0:
                    print(f"[gaze-box] Epoch {epoch+1} Step {step+1}/{FLAGS.steps_per_epoch}: "
                          f"total={float(total):.4f} ce={float(ce):.4f} regL1_box={float(reg):.4f} "
                          f"acc={float(acc):.4f} box_L1={float(reg_l1):.4f}")
                # ——训练集统计器累计——
                epoch_acc_sum += float(acc)
                epoch_regl1_sum += float(reg_l1)
                epoch_cnt += 1
                step += 1
                
            # === 每个 epoch 结束：先打印训练平均 ===
            # print(f"[gaze][train-avg] Epoch {epoch+1}: acc_mean={epoch_acc_sum/epoch_cnt:.4f}, MAE_xy={epoch_regl1_sum/epoch_cnt:.4f}")
            train_acc_mean = epoch_acc_sum / epoch_cnt
            train_mae_mean = epoch_regl1_sum / epoch_cnt
            print(f"[gaze-box][train-avg] Epoch {epoch+1}: acc_mean={train_acc_mean:.4f}, MAE_box(L1)={train_mae_mean:.4f}")
            # === 再跑一次验证集 ===
            val_acc, val_mae = eval_epoch_gaze_box(state, FLAGS.gaze_data_root_val, batch_size=FLAGS.batch_size, image_key="front_camera")
            print(f"[gaze-box][val]      Epoch {epoch+1}: acc={val_acc:.4f}, MAE_box(L1)={val_mae:.4f}")
            
            if train_acc_mean > best_train["acc"]:
                best_train["acc"] = float(train_acc_mean)
                best_train["epoch"] = int(epoch + 1)
                best_train["val_at_best"] = float(val_acc)
            # —— 更新“最佳验证 acc”记录（同时记下该 epoch 的 train acc）
            if val_acc > best_val["acc"]:
                best_val["acc"] = float(val_acc)
                best_val["epoch"] = int(epoch + 1)
                best_val["train_at_best"] = float(train_acc_mean)
            # 若设置了 ckpt_every，则按间隔保存；否则只在最后保存
            if (FLAGS.ckpt_every and (epoch + 1) % FLAGS.ckpt_every == 0) or (epoch + 1 == FLAGS.num_epochs):
                ckpt_dir = os.path.join(FLAGS.ckpt_root, "gaze_ckpt")
                os.makedirs(ckpt_dir, exist_ok=True)
                checkpoints.save_checkpoint(
                    ckpt_dir,
                    state,
                    step=epoch + 1,
                    overwrite=False,
                    keep=100
                )
                print(f"[gaze-box][ckpt] Saved checkpoint at epoch {epoch+1} -> {ckpt_dir}")
            
        # —— 训练全部结束后，先做最后一次保存，再汇总打印两条最佳
        print("\n[gaze-box][summary] ===============================")
        print(f"[gaze-box][summary] Best TRAIN acc_mean: {best_train['acc']:.4f} "
              f"(epoch={best_train['epoch']}), VAL acc at that epoch: {best_train['val_at_best']:.4f}")
        print(f"[gaze-box][summary] Best VAL   acc:      {best_val['acc']:.4f} "
              f"(epoch={best_val['epoch']}), TRAIN acc_mean at that epoch: {best_val['train_at_best']:.4f}")
        print("[gaze-box][summary] ===============================\n")
        # 若未设置周期性保存（ckpt_every==0），在这里做一次最终保存
        if not FLAGS.ckpt_every:
            ckpt_dir = os.path.join(FLAGS.ckpt_root, "gaze_ckpt")
            os.makedirs(ckpt_dir, exist_ok=True)
            checkpoints.save_checkpoint(
                ckpt_dir,
                state,
                step=FLAGS.num_epochs,
                overwrite=False,
                keep=100
            )
            print(f"[gaze-box][ckpt] Saved final checkpoint at epoch {FLAGS.num_epochs} -> {ckpt_dir}")
            


    else:
        # -------- 保留原 reward 二分类训练路径（你的旧逻辑）---------
        # Create buffer for positive transitions
        pos_buffer = ReplayBuffer(
            observation_space=env.observation_space,
            action_space=env.action_space,
            capacity=10000,
            include_label=True,
        )
        if FLAGS.is_pick_and_place_task:
            success_paths = glob.glob(os.path.join(os.getcwd(), "classifier_data", "*success*.pkl"))
        elif FLAGS.is_pick_task:
            success_paths = glob.glob(os.path.join(os.getcwd(), "classifier_data_pick", "*success*.pkl"))
        else:
            success_paths = glob.glob(os.path.join(os.getcwd(), "classifier_data_place", "*success*.pkl"))

        for path in success_paths:
            success_data = []
            with open(path, "rb") as f:
                while True:
                    try:
                        success_data.extend(pkl.load(f))
                    except EOFError:
                        break
            for trans in success_data:
                trans["labels"] = 1
                pos_buffer.insert(trans)

        pos_iterator = pos_buffer.get_iterator(sample_args={"batch_size": FLAGS.batch_size // 2},
                                               device=sharding.replicate())

        # Create buffer for negative transitions
        neg_buffer = ReplayBuffer(
            observation_space=env.observation_space,
            action_space=env.action_space,
            capacity=10000,
            include_label=True,
        )
        if FLAGS.is_pick_and_place_task:
            failure_paths = glob.glob(os.path.join(os.getcwd(), "classifier_data", "*failure*.pkl"))
        elif FLAGS.is_pick_task:
            failure_paths = glob.glob(os.path.join(os.getcwd(), "classifier_data_pick", "*failure*.pkl"))
        else:
            failure_paths = glob.glob(os.path.join(os.getcwd(), "classifier_data_place", "*failure*.pkl"))

        for path in failure_paths:
            failure_data = []
            with open(path, "rb") as f:
                while True:
                    try:
                        failure_data.extend(pkl.load(f))
                    except EOFError:
                        break
            for trans in failure_data:
                trans["labels"] = 0
                neg_buffer.insert(trans)

        neg_iterator = neg_buffer.get_iterator(sample_args={"batch_size": FLAGS.batch_size // 2},
                                               device=sharding.replicate())

        print(f"failed buffer size: {len(neg_buffer)}")
        print(f"success buffer size: {len(pos_buffer)}")

        rng, key = jax.random.split(rng)
        pos_sample = next(pos_iterator); neg_sample = next(neg_iterator)
        sample = concat_batches(pos_sample, neg_sample, axis=0)

        rng, key = jax.random.split(rng)
        state = create_classifier_reward(key,
                                         sample["observations"],
                                         config.classifier_keys,
                                         image_key_weights=config.classifier_key_weights)

        def data_augmentation_fn(rng, observations):
            for pixel_key in config.classifier_keys:
                observations = observations.copy(
                    add_or_replace={
                        pixel_key: batched_random_crop(
                            observations[pixel_key], rng, padding=4, num_batch_dims=2
                        )
                    }
                )
            return observations

        @jax.jit
        def train_step(state, batch, key):
            def loss_fn(params):
                logits = state.apply_fn({"params": params},
                                        batch["observations"],
                                        rngs={"dropout": key}, train=True)
                return optax.sigmoid_binary_cross_entropy(logits, batch["labels"]).mean()
            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            new_state = state.apply_gradients(grads=grads)
            logits = new_state.apply_fn({"params": new_state.params},
                                        batch["observations"], train=False, rngs={"dropout": key})
            train_accuracy = jnp.mean((nn.sigmoid(logits) >= 0.95) == batch["labels"])
            return new_state, loss, train_accuracy

        for epoch in tqdm(range(FLAGS.num_epochs)):
            pos_sample = next(pos_iterator)
            neg_sample = next(neg_iterator)
            batch = concat_batches(pos_sample, neg_sample, axis=0)
            rng, key = jax.random.split(rng)
            obs = data_augmentation_fn(key, batch["observations"])
            batch = batch.copy(add_or_replace={"observations": obs, "labels": batch["labels"][..., None]})
            rng, key = jax.random.split(rng)
            state, train_loss, train_accuracy = train_step(state, batch, key)
            print(f"[reward] Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

        # 保存与原来一致
        if FLAGS.is_pick_and_place_task:
            outdir = "classifier_ckpt/"
        elif FLAGS.is_pick_task:
            outdir = "classifier_ckpt_pick/"
        else:
            outdir = "classifier_ckpt_place/"
        os.makedirs(outdir, exist_ok=True)
        checkpoints.save_checkpoint(outdir, state, step=FLAGS.num_epochs, overwrite=False, keep=100)

if __name__ == "__main__":
    app.run(main)
