from ..builder import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class SALT(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SALT, self).__init__(backbone, neck, bbox_head, train_cfg,
                                  test_cfg, pretrained)
