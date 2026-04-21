from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.config.mosstrack.config import cfg, update_config_from_file


def parameters(yaml_name: str, run_id=None):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    save_dir = env_settings().save_dir
    # update default config from yaml file
    yaml_file = os.path.join(prj_dir, 'experiments/mosstrack/%s.yaml' % yaml_name)
    # yaml_file = 'C:/Users/ROG/Desktop/DUTrack1/experiments/mosstrack/mosstrack_256_full.yaml'
    update_config_from_file(yaml_file)
    params.cfg = cfg
    # print("test config: ", cfg)

    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    # Network checkpoint path
    if run_id is None:
        # params.checkpoint = os.path.join(save_dir, "checkpoints/train/mosstrack/%s/DUTrack_ep%04d.pth.tar" %
        #                                 (yaml_name, cfg.TEST.EPOCH))
        params.checkpoint = "/vcl/2025liuyisong/DUTrack1/output/checkpoints/train/dutrack/dutrack_256_full_18(best)/DUTrack_ep0018.pth.tar"
    else:
        params.checkpoint = os.path.join(save_dir, "checkpoints/train/mosstrack/%s/DUTrack_ep%04d.pth.tar" %
                                        (yaml_name, run_id))
    
    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
