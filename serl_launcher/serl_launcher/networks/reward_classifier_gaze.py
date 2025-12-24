import pickle as pkl
import jax
from jax import numpy as jnp
from typing import Tuple
import flax.linen as nn
from flax.training.train_state import TrainState
from flax.training import checkpoints
import optax
from typing import Callable, Dict, List
import requests
import os
from tqdm import tqdm

from serl_launcher.vision.resnet_v1 import resnetv1_configs, PreTrainedResNetEncoder
from serl_launcher.common.encoding import EncodingWrapper

# -------------------- heads --------------------

class BinaryClassifier(nn.Module):
    encoder_def: nn.Module
    hidden_dim: int = 256
    @nn.compact
    def __call__(self, x, train=False):
        x = self.encoder_def(x, train=train)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.Dropout(0.1)(x, deterministic=not train)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x

class NWayClassifier(nn.Module):
    encoder_def: nn.Module
    hidden_dim: int = 256
    n_way: int = 3
    @nn.compact
    def __call__(self, x, train=False):
        x = self.encoder_def(x, train=train)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.Dropout(0.1)(x, deterministic=not train)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_way)(x)
        return x

class GazeBBoxHead(nn.Module):
    """三分类（0=obj0,1=obj1,2=none）+ bbox 回归；bbox 输出为 xyxy，均归一化到 [0,1]。"""
    encoder_def: nn.Module
    hidden_dim: int = 256
    num_classes: int = 3
    @nn.compact
    def __call__(self, x, train=False):
        x = self.encoder_def(x, train=train)   # (B, D)
        h = nn.Dense(self.hidden_dim)(x)
        h = nn.Dropout(0.1)(h, deterministic=not train)
        h = nn.LayerNorm()(h)
        h = nn.relu(h)
        logits = nn.Dense(self.num_classes, name="head_cls")(h)      # (B,3)
        cxcywh = nn.Dense(4, name="head_box_cxcywh")(h)              # (B,4)
        cxcywh = nn.sigmoid(cxcywh)                                  # 归一化到 [0,1]
        cx, cy, w, h_ = jnp.split(cxcywh, 4, axis=-1)
        min_size = 1e-3
        w  = jnp.clip(w,  min_size, 1.0)
        h_ = jnp.clip(h_, min_size, 1.0)
        x0 = jnp.clip(cx - 0.5 * w,  0.0, 1.0)
        y0 = jnp.clip(cy - 0.5 * h_, 0.0, 1.0)
        x1 = jnp.clip(cx + 0.5 * w,  0.0, 1.0)
        y1 = jnp.clip(cy + 0.5 * h_, 0.0, 1.0)
        box_xyxy = jnp.concatenate([x0, y0, x1, y1], axis=-1)        # (B,4)
        return logits, box_xyxy

# -------------------- builder --------------------

def _build_encoder(image_keys: List[str], image_key_weights: Dict[str, float] | None):
    pretrained_encoder = resnetv1_configs["resnetv1-10-frozen"](
        pre_pooling=True,
        name="pretrained_encoder",
    )
    encoders = {
        image_key: PreTrainedResNetEncoder(
            pooling_method="spatial_learned_embeddings",
            num_spatial_blocks=8,
            bottleneck_dim=256,
            pretrained_encoder=pretrained_encoder,
            name=f"encoder_{image_key}",
        )
        for image_key in image_keys
    }
    encoder_def = EncodingWrapper(
        encoder=encoders,
        use_proprio=False,
        enable_stacking=False,
        image_keys=image_keys,
        image_weights=image_key_weights,
    )
    return encoder_def

def _load_resnet10_into(params_tree, image_keys: List[str]):
    # ~/.serl/resnet10_params.pkl
    home = os.path.expanduser("~/.serl")
    os.makedirs(home, exist_ok=True)
    fp = os.path.join(home, "resnet10_params.pkl")
    if not os.path.exists(fp):
        url = "https://github.com/rail-berkeley/serl/releases/download/resnet10/resnet10_params.pkl"
        print(f"Downloading pretrained ResNet-10 from {url}")
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(fp, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                f.write(chunk)
        print("Download complete.")

    # with open(fp, "rb") as f:
    #     encoder_params = pkl.load(f)
    with open(fp, "rb") as f:
        encoder_params = pkl.load(f)

    from jax import tree_util as jtu
    cnt = sum(x.size for x in jtu.tree_leaves(encoder_params))
    print(f"Loaded {cnt/1e6:.6f}M parameters from ResNet-10 pretrained on ImageNet-1K")


    from flax.core import FrozenDict
    from flax.core.frozen_dict import unfreeze, freeze
    new_params = unfreeze(params_tree) if isinstance(params_tree, FrozenDict) else params_tree
    import numpy as np
    def _safe_merge(dst, src, path=()):
        if isinstance(dst, dict) and isinstance(src, dict):
            for k, v in src.items():
                if k in dst:
                    dst[k] = _safe_merge(dst[k], v, path + (k,))
            return dst
        if isinstance(dst, np.ndarray) and isinstance(src, np.ndarray):
            if dst.shape == src.shape:
                return src
            else:
                return dst
        return dst
    
    missing_reported = False
    for image_key in image_keys:
        try:
            dst = new_params["encoder_def"][f"encoder_{image_key}"]["pretrained_encoder"]
        except KeyError:
            if not missing_reported:
                print(f"[warn] pretrained_encoder scope not found under encoder_{image_key}")
                missing_reported = True
            continue
        _safe_merge(dst, encoder_params)
    
    return freeze(new_params)
from flax.core.frozen_dict import unfreeze

def create_classifier(
    key, sample, image_keys, image_key_weights=None, n_way=2, task="gaze_box",
) -> TrainState:
    encoder_def = _build_encoder(image_keys, image_key_weights)
    if task == "gaze_box":
        classifier_def = GazeBBoxHead(encoder_def=encoder_def, num_classes=3)
    else:
        classifier_def = BinaryClassifier(encoder_def=encoder_def) if n_way == 2 \
            else NWayClassifier(encoder_def=encoder_def, n_way=n_way)

    # 1) init
    params0 = classifier_def.init(key, sample)["params"]
    # 2) 合并预训练
    params1 = _load_resnet10_into(params0, image_keys)  # 返回 FrozenDict
    # 3) 统一成 dict
    params_final = unfreeze(params1)

    # 4) 再 create（opt_state 和 params 同 treedef）
    state = TrainState.create(
        apply_fn=classifier_def.apply,
        params=params_final,
        tx=optax.adam(1e-4),
    )
    return state

def _ensure_nested_sample(sample: Dict, image_keys: List[str]) -> Dict:
    if isinstance(sample, dict) and "images" in sample and isinstance(sample["images"], dict):
        return sample
    if isinstance(sample, dict) and all(k in sample for k in image_keys):
        return {"images": {k: sample[k] for k in image_keys}}
    raise KeyError(
        f"Sample does not contain required image keys {image_keys}. "
        f"Got top-level keys: {list(sample.keys()) if isinstance(sample, dict) else type(sample)}"
    )

# -------------------- inference helpers --------------------

def load_classifier_func(
    key: jnp.ndarray,
    sample: Dict,
    image_keys: List[str],
    checkpoint_path: str,
    image_key_weights: Dict[str, float] | None = None,
    n_way: int = 2,
) -> Callable[[Dict], jnp.ndarray]:
    """旧：reward 二分类/多分类用。"""
    sample = _ensure_nested_sample(sample, image_keys)
    state = create_classifier(key, sample, image_keys, image_key_weights, n_way=n_way, task="reward")
    # state = create_classifier(key, sample, image_keys, image_key_weights, n_way=n_way, task="reward")
    state = checkpoints.restore_checkpoint(checkpoint_path, target=state)
    fn = lambda obs: state.apply_fn({"params": state.params}, obs, train=False)
    return jax.jit(fn)

def load_gaze_bbox_classifier_func(
    key: jnp.ndarray,
    sample: Dict,
    image_keys: List[str],
    checkpoint_path: str,
    image_key_weights: Dict[str, float] | None = None,
) -> Callable[[Dict], Dict[str, jnp.ndarray]]:
    sample = _ensure_nested_sample(sample, image_keys)
    state = create_classifier(key, sample, image_keys, image_key_weights, task="gaze_box")
    # state = create_classifier(key, sample, image_keys, image_key_weights, task="gaze_box")
    state = checkpoints.restore_checkpoint(checkpoint_path, target=state)
    def _fn(obs):
    #     logits, box = state.apply_fn({"params": state.params}, obs, train=False)
    #     return {"logits": logits, "gaze_box": box}
    # return jax.jit(_fn)
        logits, box = state.apply_fn({"params": state.params}, obs, train=False)
        # 统一返回为 tuple，兼容调用端 `logits, gaze_box = ...`
        return logits, box
    print("[gaze_bbox] load_gaze_bbox_classifier_func: USING TUPLE API")  # 一次性标记
    return jax.jit(_fn)
    
