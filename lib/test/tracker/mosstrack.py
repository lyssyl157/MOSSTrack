from lib.models.mosstrack import build_dutrack
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os
from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box



class MoSSTrack(BaseTracker):
    def __init__(self, params):
        super(MoSSTrack, self).__init__(params)
        network = build_dutrack(params.cfg, training=False)
        checkpoint = torch.load(self.params.checkpoint, map_location='cpu', weights_only=False)['net']
        network.load_state_dict(checkpoint, strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE

        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            # else:
            #     # self.add_hook()
            #     self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}
        self.num_template = self.cfg.TEST.TEMPLATE_NUMBER
        self.update_intervals = 50
        # self.descriptgenRefiner = descriptgenRefiner(params.cfg.MODEL.BACKBONE.BLIP_DIR,params.cfg.MODEL.BACKBONE.BERT_DIR)

    def initialize(self, image, info: dict):
        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        z_patch_arr = z_patch_arr
        self.vis = z_patch_arr
        template = self.preprocessor.process(z_patch_arr)
        self.template_list = [template] * self.num_template

        template_bbox = self.transform_image_to_crop(torch.tensor(info['init_bbox']),
                                                    torch.tensor(info['init_bbox']),
                                                    resize_factor,
                                                    torch.Tensor([self.params.template_size, self.params.template_size]),
                                                    normalize=True)
        self.template_anno_list = [template_bbox.unsqueeze(0)]
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}



    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr)
        search_list = [search]
        if self.frame_id == 1:
            self.pred_boxes = None

        with torch.no_grad():
            out_dict = self.network.forward(pred_boxes=self.pred_boxes,
                                            template_list=self.template_list,
                                            search_list=search_list,
                                            template_anno_list=self.template_anno_list,)
        if isinstance(out_dict, list):
            out_dict = out_dict[-1]
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes, conf_score = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'], return_score=True)
        pred_boxes = pred_boxes.view(-1, 4)
        self.pred_boxes = pred_boxes
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # update the template
        if self.num_template > 1:
            if (self.frame_id % self.update_intervals == 0) and (conf_score > 0.7):
                z_patch_arr, resize_factor, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                           output_sz=self.params.template_size)
                template = self.preprocessor.process(z_patch_arr)
                self.template_list.append(template)

                if len(self.template_list) > self.num_template:
                    self.template_list.pop(1)

                prev_box_crop = self.transform_bbox_to_crop(self.state, resize_factor,
                                                    template.device)
                self.template_anno_list.append(prev_box_crop.unsqueeze(0))
                if len(self.template_anno_list) > self.num_template:
                    self.template_anno_list.pop(1)

        # for debug
        if image.shape[-1] == 6:
            image_show = image[:, :, :3]
        else:
            image_show = image
        if self.debug == 1:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image_show, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
            cv2.imshow('vis', image_BGR)
            cv2.waitKey(1)

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}

    # def select_memory_frames(self):
    #     num_segments = self.cfg.TEST.TEMPLATE_NUMBER
    #     cur_frame_idx = self.frame_id
    #     if num_segments != 1:
    #         assert cur_frame_idx > num_segments
    #         dur = cur_frame_idx // num_segments
    #         indexes = np.concatenate([
    #             np.array([0]),
    #             np.array(list(range(num_segments))) * dur + dur // 2
    #         ])
    #     else:
    #         indexes = np.array([0])
    #     indexes = np.unique(indexes)
    #
    #     select_frames_RGB, select_frames_TIR, select_masks = [], [], []
    #
    #     for idx in indexes:
    #         frames_RGB = self.memory_frames_RGB[idx]
    #         frames_TIR = self.memory_frames_TIR[idx]
    #         if not frames_RGB.is_cuda:
    #             frames_RGB = frames_RGB.cuda()
    #         if not frames_TIR.is_cuda:
    #             frames_TIR = frames_TIR.cuda()
    #
    #
    #         select_frames_RGB.append(frames_RGB)
    #         select_frames_TIR.append(frames_TIR)
    #
    #         # if self.cfg.MODEL.BACKBONE.CE_LOC:
    #         #     box_mask_z = self.memory_masks[idx]
    #         #     select_masks.append(box_mask_z.cuda())
    #
    #     # if self.cfg.MODEL.BACKBONE.CE_LOC:
    #     #     return select_frames_RGB, select_frames_TIR, torch.cat(select_masks, dim=1)
    #     # else:
    #     return select_frames_RGB, select_frames_TIR, None
    
    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights

def get_tracker_class():
    return MoSSTrack
