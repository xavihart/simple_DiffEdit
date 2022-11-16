# denoise using diffusion 
import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
import torchvision
from torchvision import transforms as T, utils
import time


from datasets import get_dataset, DATASETS, get_num_classes
from PIL import Image
# from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)




def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="/home/hxue45/data/sem1/Diff-Smoothing/model/dm/256x256_diffusion_uncond.pt",
        classifier_path="",
        classifier_scale=1.0,
        sigma=0.25,
        skip=100
    )
    model_config = dict(
                    use_fp16=False,
                    attention_resolutions="32, 16, 8",
                    class_cond=False,
                    diffusion_steps=1000,
                    image_size=256,
                    learn_sigma=True,
                    noise_schedule='linear',
                    num_channels=256,
                    num_head_channels=64,
                    num_res_blocks=2,
                    resblock_updown=True,
                    use_scale_shift_norm=True,
                    img_pth=None,
                    sample_num=10,
                    process_shown=8
                )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    defaults.update(model_config)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


# --attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True

def explore(x:th.Tensor):
    print(x.shape, x.max(), x.min())

if __name__ == '__main__':
    os.environ['IMAGENET_LOC_ENV'] = "/home/hxue45/data/datasets/imagenet/"

    args = create_argparser().parse_args()
    model, diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )

    model.load_state_dict(
        th.load(args.model_path)
    )

    # model = model.to(0)  # 35000 Mb GPU


    # print(diffusion.alphas_cumprod, diffusion.alphas_cumprod.shape)

    # dataset = get_dataset('imagenet', 'test')
    

    sigma_list = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]


    
    with th.no_grad():

        image_path = f'../image_try/{args.img_pth}.png'
        img = Image.open(image_path).convert('RGB')
        transform = T.Compose([
                        T.Resize(int(224)),
                        # T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
                        T.CenterCrop(224),
                        T.ToTensor()
                    ])
        x_raw = transform(img)


        result_path = image_path[:-4] + f'/sample_{args.sample_num}/'

        if not os.path.exists(result_path):
            os.makedirs(result_path)
       

        f = open(result_path + '/out.log', "a")

        full_img = []

        time_bg = time.time()

        

        for sigma in sigma_list:
            print(sigma)
            f.write(f'{sigma:_^20}\n')

            time_sigma_begin = time.time()
            # if i % args.skip != 0:
            #     continue

            # (x, label) = dataset[i]
            # # x : (0 - 1)

            
            # use own data
            # load image

            x = x_raw
          


            x = 2 * x - 1. # scale to [-1, 1]

            noise = th.randn_like(x) * sigma

            x_noised = x + noise

            x_noised_raw = x_noised
            
            t = np.abs(diffusion.alphas_cumprod - 1 / (1 + sigma ** 2)).argmin() # 521
            
            print(t)
            # print(t_opt)
            # print(diffusion.alphas_cumprod[t_opt])

            # print(type(x_noised))
            x_noised = x_noised * np.sqrt(diffusion.alphas_cumprod[t])
            t_b = th.full((1, ), t).long()
            x_noised = x_noised[None, ...].float()
            # print(x_noised.shape)

            # cuda
            model = model.cuda()
            x_noised = x_noised.cuda()
            t_b = t_b.cuda()
            # print(x_noised.dtype, t_b.dtype)
            # denoise

            x_denoised = x_noised

            

            if args.sample_num == -1:
                sample_num = t
            else:
                sample_num = args.sample_num

            sample_img_list = [(x_denoised[0].cpu()+1)*0.5]
            
            for i_ in range(sample_num):
                
                if i_ == sample_num - 1:
                    x_denoised = diffusion.p_sample(model, x_denoised, t_b)['pred_xstart']
                else:
                    x_denoised = diffusion.p_sample(model, x_denoised, t_b)['sample']
                t_b -= 1
                
                if i_ % (sample_num // args.process_shown + 1) == 0:
                    if i_ != sample_num - 1:
                        sample_img_list.append((x_denoised[0].cpu()+1)*0.5)
            
            sample_process_img = th.cat(sample_img_list, 2)

            torchvision.utils.save_image(sample_process_img, result_path + f"/diffusion_process_sigma{sigma}.png")

            f.write(f"time cmd : {time.time() - time_sigma_begin}\n")
            
            
            # explore(x_denoised)
            
            image_show = 0.5 * (th.cat([x, x_noised_raw, x_denoised[0].cpu()], 1) + 1)


            full_img.append(image_show)

        print(f'TIME CMD:{time.time() - time_bg}')
        f.close()
        full_img = th.cat(full_img, 2)
        save_pth = result_path + f'diff_sigma.jpg'
        # if not os.path.exists(save_pth):
        #     os.makedirs(save_pth)
        torchvision.utils.save_image(full_img, save_pth)
        

            
            


 
    
    