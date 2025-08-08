import torch
import json
import os
import argparse
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from torch.nn import functional as F
from efficientvit.model_zoo import MGVQ_HF
from tqdm import tqdm
import numpy as np
from skimage.metrics import structural_similarity as ssim
from applications.tokenizer.lpips import LPIPS
from PIL import Image
from cleanfid import fid
from eval_aug import resize_center_crop
# generate
parser = argparse.ArgumentParser(description="")
parser.add_argument("--vq-model", type=str, choices=['mgvq-f8c32', 'mgvq-f16c32', 'mgvq-f32c32'], default="mgvq-f16c32") 
parser.add_argument("--vq-ckpt", type=str, default='/path/to/your/ckpt', help="ckpt path for vq model") 
parser.add_argument("--codebook-size", type=int, default=32768, help="codebook size for vector quantization")
parser.add_argument("--codebook-dim", type=int, default=32, help="codebook dimensions for vector quantization")
parser.add_argument("--codebook-groups", type=int, default=4, help="codebook groups for vector quantization")
parser.add_argument("--groups-to-use", type=int, default=4, help="codebook groups used for vector quantization")
parser.add_argument("--eval-dataset", type=str, default="imagenet256p", help="benchmark for evaluation")
parser.add_argument("--ds-rate", type=int, default=16, help="downsample ratio")
parser.add_argument("--path-to-save", type=str, default="./eval_imgs", help="path to save evaluation images")
parser.add_argument("--dataset-root", type=str, default="/path/to/your/dataset", help="path to evaluation datasets")
parser.add_argument("--eval-fid", action='store_true', help="weither to save images and eval rfid")

args = parser.parse_args()

def load_vqgan(args):
    # create and load model
    vq_model = MGVQ_HF(args)
    vq_model.cuda()
    vq_model.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    return vq_model

ds_rate = args.ds_rate
dir_name = args.vq_ckpt.split('/')[-1].split('.')[-2]
gt_dir = os.path.join(args.path_to_save, f'gt/{args.eval_dataset}_imgs')
fake_dir = os.path.join(args.path_to_save, f'{dir_name}/{args.eval_dataset}_imgs')
if args.eval_fid:
    save_gt_imgs = True
    save_fake_imgs = True
    if not os.path.exists(gt_dir):
        os.makedirs(gt_dir)
    if not os.path.exists(fake_dir):    
        os.makedirs(fake_dir)
else:
    save_gt_imgs = False
    save_fake_imgs = False


vqvae = load_vqgan(args)

mse_mean = 0
ssim_mean = 0
lpips_mean = 0
lmodel = LPIPS().cuda()
img_path_data = []

if args.eval_dataset == "imagenet256p":
    test_res_w = 256
    test_res_h = 256
    root_path = args.dataset_root # .../origin/val
    for seq in tqdm(os.listdir(root_path)):
        if os.path.isdir(os.path.join(root_path, seq)):
            for img in os.listdir(os.path.join(root_path, seq)):
                img_path_data.append(os.path.join(root_path, seq, img))

elif args.eval_dataset == "UHDBench2k":
    test_res_w = 2560
    test_res_h = 1440
    json_path = os.path.join(args.dataset_root, 'UHDBench.json') # .../UHDBench
    with open(json_path, 'r') as file:
        json_data = json.load(file)
    for seq in json_data:
        for img in json_data[seq]:
            img_path_data.append(os.path.join(args.dataset_root, img))


length = len(img_path_data)
print(f"len:{length}")
for i in tqdm(range(0, length)):
    im = resize_center_crop(img_path_data[i], test_res_w, test_res_h)
    h, w, _ = im.shape
    width = int( (w + ds_rate-1) // ds_rate)  * ds_rate
    height = int( (h + ds_rate-1) // ds_rate)  * ds_rate

    if save_gt_imgs:
        save_path_img = img_path_data[i].split('/')[-1]
        save_path_name = save_path_img.split('.')[0]
        im_pil = Image.fromarray(im)
        im_pil.save(os.path.join(gt_dir, f'{i}_{save_path_name}.png'))
    
    im_tensor = (torch.from_numpy(im.copy()).permute(2, 0, 1).unsqueeze(0).cuda()/255.-0.5)*2

    with torch.no_grad():
        quant, diff, info = vqvae.encode(im_tensor)
        qzshape = [1, args.codebook_dim, height//ds_rate, width//ds_rate]
        image = vqvae.decode_code(info[2].unsqueeze(0).reshape(-1,1,args.codebook_groups), qzshape, groups_to_use=args.groups_to_use) # output value is between [-1, 1]

    # eval
    image_eval = image[0].clip(-1,1)*0.5+0.5
    im_eval = im_tensor[0].clip(-1,1)*0.5+0.5
    tmp_mse = F.mse_loss(image_eval, im_eval).item()
    mse_mean += tmp_mse
    image_u8 = ((image_eval.permute(1,2,0).detach().cpu().numpy())*255).astype('uint8')
    im_u8 = ((im_eval.permute(1,2,0).detach().cpu().numpy())*255).astype('uint8')
    tmp_ssim = ssim(image_u8, im_u8, full=False, channel_axis=-1)
    ssim_mean += tmp_ssim
    tmp_lpips = lmodel(image_eval, im_eval).item()
    lpips_mean += tmp_lpips

    if save_fake_imgs:
        image = (image / 2 + 0.5).clip(0, 1) 
        im = (image[0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype('uint8')
        im_pil = Image.fromarray(im)
        im_pil.save(os.path.join(fake_dir, f'{i}_{save_path_name}.png'))

mse_mean /= length
ssim_mean /= length
lpips_mean /= length
psnr_clip = 20 * np.log10(1.0/np.sqrt(mse_mean))

print(f"dataset: {args.eval_dataset}, ckpt: {args.vq_ckpt}")
print(f"psnr: {psnr_clip}, ssim: {ssim_mean}, lpips: {lpips_mean}")

if args.eval_fid:
    model_name = "inception_v3"
    fid_value = fid.compute_fid(gt_dir,
                            fake_dir,
                            model_name=model_name)
    print(f"fid: {fid_value}")