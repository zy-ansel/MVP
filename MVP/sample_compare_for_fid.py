import os
import os.path as osp
import torch, torchvision
import random
import numpy as np
from glob import glob
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
from PIL import Image
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
import sys
from models import VQVAE, build_vae_var
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image

MODEL_DEPTH = 20
assert MODEL_DEPTH in {16, 20, 24, 30}

vae_ckpt = 'ckpt/var/vae_ch160v4096z32.pth'
mvp_ckpt = f'VAR/log2/local_output_c2i_d{MODEL_DEPTH}_28/ar-ckpt-best.pth'
var_ckpt = f'/ckpt/var/var_d{MODEL_DEPTH}.pth'

patch_nums = (1,2,3,4,5,6,8,10,13,16)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

vae, mvp = build_vae_var(
    V=4096, Cvae=32, ch=160, share_quant_resi=4,
    device=device, patch_nums=patch_nums,
    num_classes=1000, depth=MODEL_DEPTH,
    shared_aln=False, outer_nums=28, control_strength=0.5
)
vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
mvp_ckpt = torch.load(mvp_ckpt, map_location='cpu')
mvp_wo_ddp_state = mvp_ckpt['trainer']['var_wo_ddp']
mvp.load_state_dict(mvp_wo_ddp_state, strict=True)
vae.eval(), mvp.eval()
for p in vae.parameters(): p.requires_grad_(False)
for p in mvp.parameters(): p.requires_grad_(False)
print(f'mvp prepare finished.')

def stat(name, t):
    t = t.detach().float().cpu()
    print(name, 'min/max/mean =', t.min().item(), t.max().item(), t.mean().item())

def chw_float_to_pil(x_3hw: torch.Tensor):
    x = x_3hw.detach().cpu().float()
    if x.min() < -0.01 or x.max() > 1.01:
        x = x * 0.5 + 0.5
    x = x.clamp(0, 1)
    return to_pil_image(x)

seed = 1
torch.manual_seed(seed)
num_sampling_steps = 250
cfg = 1.3 
num_classes = 1000
num_per_class = 50
class_labels = torch.arange(1000)

more_smooth = False
torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

tf32 = True
torch.backends.cudnn.allow_tf32 = bool(tf32)
torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
torch.set_float32_matmul_precision('high' if tf32 else 'highest')

# === CHANGED === 开关：需要的话可以一键关闭 autocast 做对照
USE_FP16_AUTOMATIC_CAST = True  # 先保留 True；如果想做数值对照，改成 False
TARGET_PER_CLASS = 50

base_dir = f"/fs/scratch/PAS2473/MM2025/neurpis2025/results_sample/mvp_d{MODEL_DEPTH}28-cfg1.3"
var_out_root = osp.join(base_dir, 'var')
mvp_out_root = osp.join(base_dir, 'mvp')
os.makedirs(var_out_root, exist_ok=True)
os.makedirs(mvp_out_root, exist_ok=True)

B = 25
for cls in tqdm(class_labels):
    cls_idx = int(cls.item())
    var_class_dir = osp.join(var_out_root, f'{cls_idx:03d}')
    mvp_class_dir = osp.join(mvp_out_root, f'{cls_idx:03d}')
    os.makedirs(var_class_dir, exist_ok=True)
    os.makedirs(mvp_class_dir, exist_ok=True)
    
    done = len(glob(osp.join(mvp_class_dir, "*.png")))
    if done >= TARGET_PER_CLASS:
        print(f"this class-({cls}) is full")
        if done > TARGET_PER_CLASS:
            print(done)
        continue

    for i in range(50 // B):
        label_B = torch.tensor([cls] * B, device=device)
        with torch.inference_mode():
            with torch.autocast("cuda", enabled=True, dtype=torch.float16, cache_enabled=True):  
                img_mvp = mvp.autoregressive_infer_cfg(
                    B,
                    label_B=label_B,
                    cfg=cfg,
                    top_k=900,
                    top_p=0.95,
                    more_smooth=more_smooth,
                    g_seed=int(seed + cls * (50 // B) + i),
                )
            img_mvp = img_mvp.permute(0, 2, 3, 1).mul_(255).cpu().numpy()
        img_mvp = img_mvp.astype(np.uint8)
        for j in range(B):
            img = PImage.fromarray(img_mvp[j])
            img.save(f"{mvp_class_dir}/{(cls * 50 + i * B + j):06d}.png")
            if j == 24:
                print(f"save in {mvp_class_dir}")

def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples, label = [], []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
        label.append(i // 50)
    samples = np.stack(samples)
    label = np.asarray(label)
    p = np.random.permutation(num)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    # np.savez(npz_path, samples=samples[p], label=label[p])
    np.savez(npz_path, arr_0=samples[p])
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path

# create_npz_from_sample_folder(var_out_root)
# create_npz_from_sample_folder(mvp_out_root)
            
        
