import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head

from lib.models.mosstrack.itpn import fast_itpn_base_3324_patch16_224
from lib.utils.box_ops import box_xyxy_to_cxcywh
# from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, \
#     XGradCAM, EigenCAM, EigenGradCAM, LayerCAM, FullGrad
# from pytorch_grad_cam import GuidedBackpropReLUModel
# from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
# import cv2
# import numpy as np
# from PIL import Image
class MoSSTrack(nn.Module):
    """ This is the base class for MMTrack """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER", token_len=1):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

        self.track_query_RGB = None
        self.track_query_TIR = None
        self.token_len = token_len

    def forward(self, template_list: torch.Tensor,
                search_list: torch.Tensor,
                template_anno_list,
                pred_boxes,
                ):

        xt_data=[]
        if not self.training:
            pred_boxes = [pred_boxes]
        for i in range(len(search_list)):
            x_RGB, x_TIR, aux_dict = self.backbone(template_list=template_list,
                                                   search_list=search_list[i],
                                                   template_anno_list=template_anno_list,
                                                   pred_boxes=pred_boxes[i],
                                                   temporal_query_RGB=self.track_query_RGB,
                                                   temporal_query_TIR=self.track_query_TIR)
            feat_last_RGB = x_RGB
            feat_last_TIR = x_TIR
            if isinstance(x_RGB, list):
                feat_last_RGB = x_RGB[-1]
                feat_last_TIR = x_TIR[-1]

            if self.backbone.add_cls_token:
                self.track_query_RGB = (aux_dict['temproal_token_RGB'].clone()).detach()  # stop grad  (B, N, C)
                self.track_query_TIR = (aux_dict['temporal_token_TIR'].clone()).detach()  # stop grad  (B, N, C)

            enc_opt_RGB = feat_last_RGB[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
            att_RGB = torch.matmul(enc_opt_RGB, x_RGB[:, :1].transpose(1, 2))  # (B, HW, N)
            opt_RGB = (enc_opt_RGB.unsqueeze(-1) * att_RGB.unsqueeze(-2)).permute((0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)


            enc_opt_TIR = feat_last_TIR[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
            att_TIR = torch.matmul(enc_opt_TIR, x_TIR[:, :1].transpose(1, 2))  # (B, HW, N)
            opt_TIR = (enc_opt_TIR.unsqueeze(-1) * att_TIR.unsqueeze(-2)).permute((0, 3, 2, 1)).contiguous()
            opt = opt_RGB + opt_TIR
            xt_data.append(opt)

            # Forward head

        xt_data = torch.cat(xt_data, dim=0)
        out = self.forward_head(xt_data, None)

        out.update(aux_dict)
        out['backbone_feat_RGB'] = x_RGB
        out['backbone_feat_TIR'] = x_TIR

        return out

    def forward_head(self, opt, gt_score_map=None):
        # run the center head
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)
        score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
        outputs_coord = bbox
        outputs_coord_new = outputs_coord.view(bs, Nq, 4)

        out = {'pred_boxes': outputs_coord_new,
               'score_map': score_map_ctr,
               'size_map': size_map,
               'offset_map': offset_map
               }

        return out

class GradCAMWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        # x 是单个 tensor 或 tuple，这里解包成模型需要的多个输入
        template_list, search_list, template_anno_list, descript, query_RGB, query_TIR = x
        out, _, _ = self.model(template_list=template_list,
                               search_list=search_list,
                               template_anno_list=template_anno_list,
                               l=descript,
                               temporal_query_RGB=query_RGB,
                               temporal_query_TIR=query_TIR)
        # GradCAM 期望返回 shape (B, C, H, W) 或 logits
        return out[-1] if isinstance(out, list) else out

def build_dutrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')

    if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'itpn_base':
        backbone = fast_itpn_base_3324_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,bert_dir=cfg.MODEL.BACKBONE.BERT_DIR)
    else:
        raise NotImplementedError

    hidden_dim = backbone.embed_dim
    patch_start_index = 1
    
    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)



    model = MoSSTrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        token_len=cfg.MODEL.BACKBONE.TOP_K,
    )
    if 'DUTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
        pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
        file_name = cfg.MODEL.PRETRAIN_FILE
        pth = os.path.join(pretrained_path,file_name)
        # pth = '/vcl/2025liuyisong/DUTrack1/output/checkpoints/train/mosstrack/best/DUTrack_ep0014(best).pth.tar'
        checkpoint = torch.load(pth, map_location="cpu", weights_only=False)
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)
    return model
