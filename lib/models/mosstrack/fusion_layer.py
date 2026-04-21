import torch
import torch.nn as nn
from lib.models.layers.attn import Attention_fusion_t2x, Attention_fusion_t2z
from timm.models.layers import Mlp, DropPath

class FusionLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.t_fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU())
        self.t2x_att = Attention_fusion_t2x(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                            proj_drop=drop)
        self.t2z_att = Attention_fusion_t2z(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                            proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x_v, x_i, temporal_query_RGB, temporal_query_TIR, lens_z):
        x_rgb = x_v[:, lens_z:, :]
        x_t = x_i[:, lens_z:, :]

        z_rgb = x_v[:, :lens_z, :]
        z_t = x_i[:, :lens_z, :]

        fused_temporal_query = torch.cat([temporal_query_RGB, temporal_query_TIR], dim=2)
        fused_temporal_query = self.t_fusion(fused_temporal_query)

        x_r_fusion, x_t_fusion = self.t2x_att(self.norm1(x_rgb), self.norm1(x_t), self.norm1(fused_temporal_query))

        x_r_fusion = x_rgb + self.drop_path(x_r_fusion)
        x_r_fusion = x_r_fusion + self.drop_path(self.mlp(self.norm2(x_r_fusion)))

        x_t_fusion = x_t + self.drop_path(x_t_fusion)
        x_t_fusion = x_t_fusion + self.drop_path(self.mlp(self.norm2(x_t_fusion)))



        z_r_fusion, z_t_fusion = self.t2z_att(self.norm1(z_rgb), self.norm1(z_t), self.norm1(fused_temporal_query))

        z_r_fusion = z_rgb + self.drop_path(z_r_fusion)
        z_r_fusion = z_r_fusion + self.drop_path(self.mlp(self.norm2(z_r_fusion)))

        z_t_fusion = z_t + self.drop_path(z_t_fusion)
        z_t_fusion = z_t_fusion + self.drop_path(self.mlp(self.norm2(z_t_fusion)))



        return x_r_fusion, x_t_fusion, z_r_fusion, z_t_fusion

