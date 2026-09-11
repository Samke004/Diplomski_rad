# Čita main.py i popravlja build_loaders funkciju
content = open('main.py').read()

old = '''    shared = dict(
        depth_dir = cfg.TRAIN_DEPTH_DIR   if cfg.USE_DEPTH     else None,
        occ_dir   = cfg.TRAIN_OCC_DIR     if cfg.USE_OCCLUSION else None,
        img_size  = (cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH),
    )

    train_ds = AppleTreeDataset(
        img_dir=cfg.TRAIN_IMG_DIR, mask_dir=cfg.TRAIN_MASK_DIR,
        transform=train_tf, **shared,
    )
    val_ds = AppleTreeDataset(
        img_dir=cfg.VAL_IMG_DIR, mask_dir=cfg.VAL_MASK_DIR,
        depth_dir=cfg.VAL_DEPTH_DIR   if cfg.USE_DEPTH     else None,
        occ_dir  =cfg.VAL_OCC_DIR     if cfg.USE_OCCLUSION else None,
        transform=val_tf,
        img_size=(cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH),
    )'''

new = '''    train_ds = AppleTreeDataset(
        img_dir=cfg.TRAIN_IMG_DIR, mask_dir=cfg.TRAIN_MASK_DIR,
        transform=train_tf,
        img_size=(cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH),
    )
    val_ds = AppleTreeDataset(
        img_dir=cfg.VAL_IMG_DIR, mask_dir=cfg.VAL_MASK_DIR,
        transform=val_tf,
        img_size=(cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH),
    )'''

if old in content:
    content = content.replace(old, new)
    open('main.py', 'w').write(content)
    print("OK — build_loaders popravljen")
else:
    print("NIJE PRONAĐENO — paste main.py build_loaders dio")
