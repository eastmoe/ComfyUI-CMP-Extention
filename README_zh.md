# ComfyUI-CMP-Extention

## 简介
ComfyUI-CMP-Extention 是一个专为 **170HX显卡** 设计的 ComfyUI 加速扩展插件。它基于 **["Instruction-Level Performance Analysis and Optimization Strategies for Constrained AI Accelerators A Case Study of CMP 170HX"](https://github.com/eastmoe/cmp_ext/blob/main/paper/paper_20260208.pdf).** 的研究成果，通过优化特定算子来提升推理性能。本扩展需要与 [cmp_ext](https://github.com/eastmoe/cmp_ext) 库配合使用。

## 安装方法

1. 确保您已安装并配置好 [cmp_ext](https://github.com/eastmoe/cmp_ext) 。
2. 进入 ComfyUI 的 `custom_nodes` 目录。
3. 执行以下命令克隆本扩展：

```bash
git clone https://github.com/eastmoe/ComfyUI-CMP-Extention
```

4. 重启 ComfyUI。

## 使用方法

1. 在 ComfyUI 中，您可以在节点菜单的 **CmpExt3** 分类下找到 **CmpExt3ControlPanelNode** 节点。
2. 将该节点插入到 **加载模型之后** 与 **采样器之前** 的位置。
3. 根据需要勾选/取消勾选要优化的算子类型。
4. 连接节点：
   - 将模型加载节点的输出连接到 `model` 输入（可选）
   - 将 CLIP 加载节点的输出连接到 `clip` 输入（可选）
   - 将 VAE 加载节点的输出连接到 `vae` 输入（可选）

## 节点参数说明

### 核心算子
- **Ops: Linear**: 线性层优化 (默认启用)
- **Ops: Conv2d**: 卷积层优化 (默认启用)
- **Ops: ConvTranspose2d (VAE Upscale)**: 转置卷积优化，用于VAE上采样 (默认启用)
- **Ops: BMM (Attention)**: 注意力机制中的批矩阵乘法优化 (默认启用)

### 归一化层
- **Norm: GroupNorm (VAE)**: 组归一化优化，主要用于VAE (默认启用)
- **Norm: LayerNorm (CLIP/Transformer)**: 层归一化优化，用于CLIP/Transformer (默认启用)

### 激活函数
- **Act: SiLU / Swish (VAE Main Suspect!)**: SiLU/Swish激活函数优化，VAE中的主要优化对象 (默认启用)
- **Act: GELU (CLIP/UNet)**: GELU激活函数优化，用于CLIP/UNet (默认启用)
- **Act: Softmax (Attention)**: Softmax激活函数优化，用于注意力机制 (默认启用)
- **Act: Mish**: Mish激活函数优化 (默认禁用)
- **Act: Softplus**: Softplus激活函数优化 (默认禁用)
- **Act: Softsign**: Softsign激活函数优化 (默认禁用)

## 注意事项

1. 本扩展专门针对 **170HX显卡** 优化，其他显卡型号可能无法获得相同效果或出现兼容性问题。
2. 如果遇到性能下降或不稳定问题，可以尝试禁用部分算子优化。


## 节点定义参考

```python
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
```

## 许可证

MIT License

---

**免责声明**：本扩展基于学术研究实现，工程稳定性有限，作者不对使用本扩展造成的任何问题负责。请在使用前充分测试并理解相关风险。
