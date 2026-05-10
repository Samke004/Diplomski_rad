import albumentations as A

def get_train_transforms(height: int = 640, width: int = 640) -> A.Compose:
    return A.Compose(
        [
            A.Resize(height, width),
            # ── Geometrijske — primjenjuju se na RGB + Depth + Mask identično ──
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1,
                rotate_limit=15, border_mode=0, p=0.5,
            ),
            A.ElasticTransform(alpha=120, sigma=6.0, p=0.3),
            A.GridDistortion(p=0.2),
            # ── Photometric — primjenjuju se SAMO na RGB (ne na depth) ──────────
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.CLAHE(clip_limit=4.0, p=0.2),
            # HueSaturationValue IZBAČEN — ne radi ispravno na additional_targets
        ],
        additional_targets={"depth": "image"},
    )

def get_val_transforms(height: int = 640, width: int = 640) -> A.Compose:
    return A.Compose(
        [A.Resize(height, width)],
        additional_targets={"depth": "image"},
    )
