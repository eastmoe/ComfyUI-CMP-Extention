# Comfyui扩展
import torch
import torch.nn.functional as F
import comfy.ops
import cmpext3

# =================================================================
# 1. 原始算子备份 (建立基准)
# =================================================================
# 我们在模块加载时就备份好原始的 PyTorch 函数
_ORIGINAL_OPS = {
    # 基础运算
    "linear": F.linear,
    "conv2d": F.conv2d,
    "conv_transpose2d": F.conv_transpose2d,
    "bmm": torch.bmm,
    
    # 归一化
    "group_norm": F.group_norm,
    "layer_norm": F.layer_norm,
    
    # 激活函数
    "silu": F.silu,
    "gelu": F.gelu,
    "softmax": F.softmax,
    "mish": F.mish,
    "softplus": F.softplus,
    "softsign": F.softsign,
}

# =================================================================
# 2. 高健壮性 Wrappers (防止内存不连续导致的 Crash)
# =================================================================

def ensure_contiguous(tensor):
    if tensor is None: return None
    if not tensor.is_contiguous():
        return tensor.contiguous()
    return tensor

# --- Linear & Conv ---
def cmpext3_linear_wrapper(input, weight, bias=None):
    is_multidim = input.dim() > 2
    original_shape = input.shape
    x = ensure_contiguous(input)
    w = ensure_contiguous(weight)
    if is_multidim: x = x.reshape(-1, original_shape[-1])
    
    if bias is None:
        b = torch.zeros(w.shape[0], device=x.device, dtype=x.dtype)
    else:
        b = ensure_contiguous(bias)

    output = cmpext3.linear(x, w, b)
    if is_multidim: output = output.view(original_shape[:-1] + (output.shape[-1],))
    return output

def cmpext3_conv2d_wrapper(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    b = ensure_contiguous(bias) if bias is not None else torch.zeros(weight.shape[0], device=input.device, dtype=input.dtype)
    return cmpext3.conv2d(ensure_contiguous(input), ensure_contiguous(weight), b, stride, padding, dilation, groups)

def cmpext3_conv_transpose2d_wrapper(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    b = ensure_contiguous(bias) # 允许为 None，视 C++ 实现而定
    return cmpext3.conv_transpose2d(ensure_contiguous(input), ensure_contiguous(weight), b, stride, padding, output_padding, dilation, groups)

def cmpext3_bmm_wrapper(input, mat2, *, out=None):
    return cmpext3.bmm(ensure_contiguous(input), ensure_contiguous(mat2))

# --- Norms ---
def cmpext3_group_norm_wrapper(input, num_groups, weight=None, bias=None, eps=1e-5):
    return cmpext3.group_norm(ensure_contiguous(input), num_groups, ensure_contiguous(weight), ensure_contiguous(bias), eps)

def cmpext3_layer_norm_wrapper(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    return cmpext3.layer_norm(ensure_contiguous(input), normalized_shape, ensure_contiguous(weight), ensure_contiguous(bias), eps)

# --- Activations (重点排查区域) ---
def cmpext3_silu_wrapper(input, inplace=False): 
    return cmpext3.silu(ensure_contiguous(input))

def cmpext3_gelu_wrapper(input, approximate='none'): 
    return cmpext3.gelu(ensure_contiguous(input))

def cmpext3_softmax_wrapper(input, dim=None, _stacklevel=3, dtype=None):
    if dtype is not None: input = input.to(dtype=dtype)
    if dim is None: dim = -1
    return cmpext3.softmax(ensure_contiguous(input), dim, False)

def cmpext3_mish_wrapper(input, inplace=False): 
    return cmpext3.mish(ensure_contiguous(input))

def cmpext3_softplus_wrapper(input, beta=1, threshold=20): 
    return cmpext3.softplus(ensure_contiguous(input), beta, threshold)

def cmpext3_softsign_wrapper(input): 
    return cmpext3.softsign(ensure_contiguous(input))


# =================================================================
# 3. 核心 Patch 管理器
# =================================================================

def apply_patch(target_obj, target_name, original_func, custom_wrapper, enable):
    """
    通用 Patch 函数：
    如果 enable 为 True，则将 target_obj.target_name 指向 custom_wrapper
    如果 enable 为 False，则将 target_obj.target_name 还原为 original_func
    """
    current_func = getattr(target_obj, target_name)
    
    if enable:
        if current_func != custom_wrapper:
            setattr(target_obj, target_name, custom_wrapper)
            # print(f"  [ON]  {target_name}")
    else:
        if current_func != original_func:
            setattr(target_obj, target_name, original_func)
            # print(f"  [OFF] {target_name}")

def update_global_patches(config):
    print(f"\n[CmpExt3 Debugger] Syncing operators...")
    
    # 1. 基础运算
    apply_patch(F, "linear", _ORIGINAL_OPS["linear"], cmpext3_linear_wrapper, config['linear'])
    apply_patch(F, "conv2d", _ORIGINAL_OPS["conv2d"], cmpext3_conv2d_wrapper, config['conv2d'])
    apply_patch(F, "conv_transpose2d", _ORIGINAL_OPS["conv_transpose2d"], cmpext3_conv_transpose2d_wrapper, config['conv_transpose2d'])
    apply_patch(torch, "bmm", _ORIGINAL_OPS["bmm"], cmpext3_bmm_wrapper, config['bmm'])
    
    # 2. 归一化
    apply_patch(F, "group_norm", _ORIGINAL_OPS["group_norm"], cmpext3_group_norm_wrapper, config['group_norm'])
    apply_patch(F, "layer_norm", _ORIGINAL_OPS["layer_norm"], cmpext3_layer_norm_wrapper, config['layer_norm'])
    
    # 3. 激活函数 (细粒度)
    apply_patch(F, "silu", _ORIGINAL_OPS["silu"], cmpext3_silu_wrapper, config['silu'])
    apply_patch(F, "gelu", _ORIGINAL_OPS["gelu"], cmpext3_gelu_wrapper, config['gelu'])
    apply_patch(F, "softmax", _ORIGINAL_OPS["softmax"], cmpext3_softmax_wrapper, config['softmax'])
    apply_patch(F, "mish", _ORIGINAL_OPS["mish"], cmpext3_mish_wrapper, config['mish'])
    apply_patch(F, "softplus", _ORIGINAL_OPS["softplus"], cmpext3_softplus_wrapper, config['softplus'])
    apply_patch(F, "softsign", _ORIGINAL_OPS["softsign"], cmpext3_softsign_wrapper, config['softsign'])

    # 特殊处理：torch.softmax 别名
    if hasattr(torch, 'softmax'):
        apply_patch(torch, "softmax", _ORIGINAL_OPS["softmax"], cmpext3_softmax_wrapper, config['softmax'])

    print("[CmpExt3 Debugger] Update complete.")


# =================================================================
# 4. ComfyUI 节点定义 (Control Panel)
# =================================================================

class CmpExt3ControlPanelNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # --- Core ---
                "enable_linear": ("BOOLEAN", {"default": True, "label": "Ops: Linear"}),
                "enable_conv2d": ("BOOLEAN", {"default": True, "label": "Ops: Conv2d"}),
                "enable_conv_transpose": ("BOOLEAN", {"default": True, "label": "Ops: ConvTranspose2d (VAE Upscale)"}),
                "enable_bmm": ("BOOLEAN", {"default": True, "label": "Ops: BMM (Attention)"}),
                
                # --- Norms ---
                "enable_group_norm": ("BOOLEAN", {"default": True, "label": "Norm: GroupNorm (VAE)"}),
                "enable_layer_norm": ("BOOLEAN", {"default": True, "label": "Norm: LayerNorm (CLIP/Transformer)"}),
                
                # --- Activations (Suspects) ---
                "enable_silu": ("BOOLEAN", {"default": True, "label": "Act: SiLU / Swish (VAE Main Suspect!)"}),
                "enable_gelu": ("BOOLEAN", {"default": True, "label": "Act: GELU (CLIP/UNet)"}),
                "enable_softmax": ("BOOLEAN", {"default": True, "label": "Act: Softmax (Attention)"}),
                "enable_mish": ("BOOLEAN", {"default": False, "label": "Act: Mish"}),
                "enable_softplus": ("BOOLEAN", {"default": False, "label": "Act: Softplus"}),
                "enable_softsign": ("BOOLEAN", {"default": False, "label": "Act: Softsign"}),
            },
            "optional": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "execute_patch"
    CATEGORY = "CmpExt3"

    def execute_patch(self, 
                      enable_linear, enable_conv2d, enable_conv_transpose, enable_bmm,
                      enable_group_norm, enable_layer_norm,
                      enable_silu, enable_gelu, enable_softmax, enable_mish, enable_softplus, enable_softsign,
                      model=None, clip=None, vae=None):
        
        config = {
            "linear": enable_linear,
            "conv2d": enable_conv2d,
            "conv_transpose2d": enable_conv_transpose,
            "bmm": enable_bmm,
            "group_norm": enable_group_norm,
            "layer_norm": enable_layer_norm,
            "silu": enable_silu,
            "gelu": enable_gelu,
            "softmax": enable_softmax,
            "mish": enable_mish,
            "softplus": enable_softplus,
            "softsign": enable_softsign
        }
        
        update_global_patches(config)
        
        return (model, clip, vae)

# 注册节点
NODE_CLASS_MAPPINGS = {
    "CmpExt3ControlPanel": CmpExt3ControlPanelNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CmpExt3ControlPanel": "CmpExt3 Full Control Panel"
}
