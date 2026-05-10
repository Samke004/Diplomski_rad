from .. import *
# Dataset config
C.dataset_name = "BranchDataset"
C.dataset_path = osp.join(C.root_dir, "BranchDataset")
C.rgb_root_folder = osp.join(C.dataset_path, "RGB")
C.rgb_format = ".png"
C.gt_root_folder = osp.join(C.dataset_path, "Label")
C.gt_format = ".png"
C.gt_transform = False
C.x_root_folder = osp.join(C.dataset_path, "Depth")
C.x_format = ".png"
C.x_is_single_channel = True
C.train_source = osp.join(C.dataset_path, "train.txt")
#C.eval_source = osp.join(C.dataset_path, "val.txt")
C.eval_source = osp.join(C.dataset_path, "test.txt")
C.is_test = False
C.num_train_imgs = 500
#C.num_eval_imgs = 80
C.num_eval_imgs = 77
C.num_classes = 4
C.class_names = ["ostalo", "deblo", "grane", "potpora"]
C.background = 255
C.image_height = 480
C.image_width = 640
C.norm_mean = np.array([0.485, 0.456, 0.406])
C.norm_std = np.array([0.229, 0.224, 0.225])