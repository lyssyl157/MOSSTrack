import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_attention1(image, attn, patch_size=16, token_index=None):
    """
    image: numpy 原图 (H, W, 3)
    attn: cross-attention map (N_q, N_k)，比如 (256, 24)
    patch_size: patch 大小
    token_index: 选择哪个 fusion token 来可视化 (0~N_k-1)
    """
    attn = attn[0]
    H, W = image.shape[:2]
    n_rows, n_cols = H // patch_size, W // patch_size

    if token_index is None:
        # 默认取注意力最大的 fusion token
        token_index = attn.mean(0).argmax()

    # 取该 fusion token 对所有图像 patch 的注意力
    attn_map = attn[:, token_index].reshape(n_rows, n_cols).detach().cpu().numpy()

    # resize 到原图大小
    attn_map_resized = cv2.resize(attn_map, (W, H), interpolation=cv2.INTER_CUBIC)

    # 归一化
    attn_map_norm = (attn_map_resized - attn_map_resized.min()) / (attn_map_resized.max() - attn_map_resized.min() + 1e-8)

    # 生成热力图
    heatmap = cv2.applyColorMap(np.uint8(255 * attn_map_norm), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # 叠加
    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

    plt.imshow(overlay)
    plt.axis("off")
    plt.show()

