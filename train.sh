CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29502 ./tools/dist_train.sh configs/panda/x101.py 8

CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29502 ./tools/dist_train.sh configs/panda/cascade_rcnn_r50_fpn_1x_coco_round1_panda.py 4

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29505 ./tools/dist_test.sh configs/panda/x101.py work_dirs/x101/latest.pth 8 --format-only --options "jsonfile_prefix=ep60"


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29502 ./tools/dist_train.sh configs/epp/epp_r50_EIOU.py 8

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29505 ./tools/dist_test.sh configs/epp/epp_r50_EIOU.py work_dirs/epp_r50_EIOU/latest.pth 8 


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29505 ./tools/dist_test.sh configs/epp/epp_r50_EIOU.py work_dirs/epp_r50_EIOU/latest.pth 8 --format-only --options "jsonfile_prefix=L1"

CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29502 ./tools/dist_train.sh configs/epp/epp_r50_EIOU.py 4 && 
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29505 ./tools/dist_test.sh configs/epp/epp_r50_EIOU.py work_dirs/epp_r50_EIOU/latest.pth 4
 
 --resume-from work_dirs/epp_r50_EIOU/latest.pth

 CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29512 ./tools/dist_train.sh configs/epp/epp_r50_EIOU_gama=1.py 4

CUDA_VISIBLE_DEVICES=0,1 PORT=29505 ./tools/dist_test.sh configs/epp/epp_r50_IOU_ms.py work_dirs/epp_r50_IOU_ms/epoch_24.pth 2 --format-only --options "jsonfile_prefix=r50-ms"

CUDA_VISIBLE_DEVICES=0,1 PORT=29505 ./tools/dist_test.sh configs/epp/epp_xt_IOU_ms.py work_dirs/epp_xt_IOU_ms/epoch_24.pth 2 --format-only --options "jsonfile_prefix=xt-ms"

&& CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29511 ./tools/dist_train.sh configs/drc/drc_r2101_DCN_fpn_2x.py 8
