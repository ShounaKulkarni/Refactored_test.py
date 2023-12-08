import argparse
import json
import os
from os.path import join

import torch
import torch.utils.data as data
import torchvision.utils as vutils

from attgan import AttGAN
from data import check_attribute_conflict, Custom, CelebA, CelebA_HQ
from helpers import Progressbar
from utils import find_model

def parse_args(args=None):
    """ Parse the command-line arguments. """
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', dest='experiment_name', required=True)
    parser.add_argument('--test_int', dest='test_int', type=float, default=1.0)
    parser.add_argument('--num_test', dest='num_test', type=int)
    parser.add_argument('--load_epoch', dest='load_epoch', type=str, default='latest')
    parser.add_argument('--custom_img', action='store_true')
    parser.add_argument('--custom_data', type=str, default='./data/custom')
    parser.add_argument('--custom_attr', type=str, default='./data/list_attr_custom.txt')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--multi_gpu', action='store_true')
    return parser.parse_args(args)

def load_settings(experiment_name):
    """ Load the experiment settings from a file. """
    with open(join('output', experiment_name, 'setting.txt'), 'r') as f:
        return json.load(f, object_hook=lambda d: argparse.Namespace(**d))

def prepare_data(args):
    """ Prepare the data for testing. """
    if args.custom_img:
        output_path = join('output', args.experiment_name, 'custom_testing')
        test_dataset = Custom(args.custom_data, args.custom_attr, args.img_size, args.attrs)
    else:
        output_path = join('output', args.experiment_name, 'sample_testing')
        if args.data == 'CelebA':
            test_dataset = CelebA(args.data_path, args.attr_path, args.img_size, 'test', args.attrs)
        elif args.data == 'CelebA-HQ':
            test_dataset = CelebA_HQ(args.data_path, args.attr_path, args.image_list_path, args.img_size, 'test', args.attrs)
    os.makedirs(output_path, exist_ok=True)
    return data.DataLoader(test_dataset, batch_size=1, num_workers=args.num_workers, shuffle=False, drop_last=False), output_path

def prepare_attributes(att_a, args):
    """ Prepare attribute combinations for testing. """
    att_b_list = [att_a]
    for i in range(args.n_attrs):
        tmp = att_a.clone()
        tmp[:, i] = 1 - tmp[:, i]
        tmp = check_attribute_conflict(tmp, args.attrs[i], args.attrs)
        att_b_list.append(tmp)
    return att_b_list

def test_model(args, test_dataloader, output_path):
    """ Test the model on the provided data loader. """
    attgan = AttGAN(args)
    attgan.load(find_model(join('output', args.experiment_name, 'checkpoint'), args.load_epoch))
    progressbar = Progressbar()

    attgan.eval()
    for idx, (img_a, att_a) in enumerate(test_dataloader):
        if args.num_test is not None and idx == args.num_test:
            break

        img_a = img_a.cuda() if args.gpu else img_a
        att_a = att_a.cuda() if args.gpu else att_a
        att_a = att_a.type(torch.float)

        att_b_list = prepare_attributes(att_a, args)
        samples = generate_images(attgan, img_a, att_b_list, args)
        save_images(samples, idx, output_path, test_dataloader.dataset, args)

def generate_images(attgan, img_a, att_b_list, args):
    """ Generate images using the model and attribute combinations. """
    with torch.no_grad():
        samples = [img_a]
        for i, att_b in enumerate(att_b_list):
            att_b_ = (att_b * 2 - 1) * args.thres_int
            if i > 0:
                att_b_[..., i - 1] = att_b_[..., i - 1] * args.test_int / args.thres_int
            samples.append(attgan.G(img_a, att_b_))
        return torch.cat(samples, dim=3)

def save_images(samples, idx, output_path, dataset, args):
    """ Save generated images to the specified path. """
    if args.custom_img:
        out_file = dataset.images[idx]
    else:
        out_file = '{:06d}.jpg'.format(idx + 182638)
    vutils.save_image(samples, join(output_path, out_file), nrow=1, normalize
