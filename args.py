import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='LAVT training and testing')
    parser.add_argument('--output-dir', default='./checkpoints/', help='path where to save checkpoint weights')
    parser.add_argument('--pin_mem', action='store_true',
                        help='If true, pin memory when using the data loader.')

    # Full model: --version default --fuse default, or don't specify any of these
    # PWAM-only: --version none --fuse default or don't specify --fuse
    # LP-only: --version default --fuse simple
    # super baseline: --version none --fuse simple

    # add
    parser.add_argument("--task_name", type=str,
                        default="SAM2-S-1024-IBLIP2-Vicun7B-Query-1-FT-QFormer-Fusion-In-Image-Pass-Next-Layer-CNN-Token-Fusion-AMP-E-40-All")
    parser.add_argument("--model_cfg", type=str, default="sam2_hiera_b+.yaml", help="model config file")
    parser.add_argument("--sam2_pretrain_path", type=str,
                        default="/home/v-zuoleili/Pretrain/sam2.1_hiera_small.pt")
    parser.add_argument("--blip2_path", type=str,
                        default="/home/v-zuoleili/Pretrain/instruct_blip_vicuna7b_trimmed.pth")
    parser.add_argument("--work_dir", type=str, default="./work_dir")
    parser.add_argument("--log_path", type=str, default="./log_path")
    parser.add_argument("--pred_save_dir", type=str, default="./pred_visual")
    # train
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--bbox_shift", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lora_rank", type=int, default=8)
    # Optimizer parameters
    parser.add_argument("--weight_decay", type=float, default=0.01, help="weight decay (default: 0.01)")
    parser.add_argument("--lr", type=float, default=1e-4, metavar="LR", help="learning rate (absolute lr), 1e-4, 6e-5")
    parser.add_argument("--mask_decoder_lr", type=float, default=3e-5, metavar="LR",
                        help="learning rate (absolute lr), 6e-5")
    parser.add_argument("--grad_accumulation_steps", type=int, default=1)
    parser.add_argument("--resume", type=str, default=None, help="Resuming training from checkpoint")
    parser.add_argument("--pretrain", type=str, default=None, help="Resuming training from checkpoint")

    parser.add_argument("--device", type=str, default="0")
    parser.add_argument('--image_size', type=int, default=1024, help='image_size')
    parser.add_argument("--cnn_img_size", type=int, default=480, help='cnn_img_size')
    parser.add_argument('--ytvos_path', type=str, default='/data/lizuolei/data/ref-youtube-vos/')
    parser.add_argument('--coco_path', type=str, default='/home/v-zuoleili/Data/coco/')
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--num_frames', default=3, type=int,
                        help="Number of clip frames for training")
    parser.add_argument('--max_skip', default=3, type=int, help="max skip frame number")

    parser.add_argument('--dataset_file', default='refcoco', help='Dataset name')

    parser.add_argument("--print_interval", type=int, default=20)
    parser.add_argument("--train_visual", action='store_true')
    parser.add_argument("--save_visual_interval", type=int, default=1000)
    parser.add_argument("--distributed", action='store_true')
    
    parser.add_argument('--refer_data_root', default='/home/v-zuoleili/Data/coco', help='REFER dataset root directory')
    parser.add_argument('--dataset', default='all', help='refcoco, refcoco+, refcocog, a2d, ytvos, joint'
                                                             'refs+ytvos')

    parser.add_argument('--split', default='train', help='only used when testing')
    parser.add_argument('--splitBy', default='unc', help='change to umd or google when the dataset is G-Ref (RefCOCOg)')

    parser.add_argument("--query_num", type=int, default=32, help="the number of query num from blip2")
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args_dict = parser.parse_args()
