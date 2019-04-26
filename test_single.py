import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import torchvision
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# python test_single.py --name flare_seg --model pix2pix --direction AtoB --results results/test --dataroot datasets/custom_flare_segs/

def tensor2PIL(img):
    img = np.transpose(img.view(-1, im_size, im_size).cpu().detach().numpy(), (1,2,0))
    img = img*im_size/2 + im_size/2
    img = Image.fromarray(img.astype('uint8'))
    return img


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    # # opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    #import pdb; pdb.set_trace()
    # dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    # root = '/home/honeywell-03/Documents/flare/flare-data-2/only-smoke/A/'
    root = opt.dataroot
    im_size = 256
    transform = transforms.Compose([
        transforms.Resize((im_size, im_size)),
        transforms.CenterCrop((im_size, im_size)),
        transforms.ToTensor()
    ])
    dataset = torchvision.datasets.ImageFolder(root, transform)
    data_loader = torch.utils.data.DataLoader(dataset)
    # import pdb; pdb.set_trace()
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    try:
        os.makedirs(opt.results_dir)
        print("Created Results Sub-directories")
    except FileExistsError:
        print("Sub-directories already exist, skipping...")

    for i,(img,_) in enumerate(data_loader):
        gen_img = model.forward_single(img)
        img = tensor2PIL(img)
        gen_img = tensor2PIL(gen_img)
        img.save(os.path.join(opt.results_dir, f'{i}_real.png'))
        gen_img.save(os.path.join(opt.results_dir, f'{i}_fake.png'))
        if i%5 == 0:
            print(f"Generating {i+1}/{len(data_loader)}", end='\r')