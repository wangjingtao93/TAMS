from model.transUNet.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from model.transUNet.networks.vit_seg_modeling import VisionTransformer as ViT_seg

def get_trans(args):
    args.vit_name = 'R50-ViT-B_16'
    args.n_skip = 3
    args.vit_patches_size = 16

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.n_classes
    config_vit.n_skip = args.n_skip
    
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.dl_resize / args.vit_patches_size), int(args.dl_resize / args.vit_patches_size))

    net = ViT_seg(config_vit, img_size=args.dl_resize, num_classes=config_vit.n_classes)

    return net