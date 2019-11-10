import os

import torch
import torch.nn.functional as F
from lnets.tasks.adversarial.mains.utils import save_image

import numpy as np

from lnets.data.load_data import load_data
from lnets.utils.config import process_config
from lnets.utils.saving_and_loading import load_model_from_config
from lnets.utils.misc import to_cuda
from lnets.models.regularization.spec_jac import jac_spectral_norm
from lnets.utils.math.autodiff import compute_jacobian

import matplotlib.pyplot as plt
from matplotlib import cm

def auc_degredation_measure(model, x, sal):
    # Compute the auc measure of the performance drop using the saliency map
    D = torch.shape(sal)

def get_saliency_map(model, x, cls):
    # The following is needed to backprop on the inputs.
    x.requires_grad = True

    # Clear the gradient buffers.
    if x.grad is not None:
        x.grad.zero_()
        for p in model.parameters():
            p.grad.zero_()

    # Take derivative of the output wrt. the inputs
    out = model(x).squeeze()[cls]
    out.backward()
    x_grad = x.grad.data
    x_grad = torch.abs(x_grad)

    return x_grad

def visualize_saliency(config):
    # Create the output directory.
    output_root = config.output_root
    if not os.path.isdir(output_root):
        os.makedirs(output_root)

    # Load a pretrained model.
    pretrained_path = config.pretrained_path
    model, pretrained_config = load_model_from_config(pretrained_path)

    # Push model to GPU if available.
    if config.cuda:
        print('Using cuda: Yes')
        model.cuda()

    model.eval()

    # Get data.
    pretrained_config.data.cuda = config.cuda
    pretrained_config.data.batch_size = config.data.batch_size
    data = load_data(pretrained_config)

    # Compute adversarial gradients and save their visualizations.
    for i, (x, y) in enumerate(data['test']):
        x = to_cuda(x, cuda=config.cuda)

        # Save the input image.
        save_path = os.path.join(output_root, 'x{}.png'.format(i))
        save_image(x, save_path)

        # Save the adversarial gradients.
        for j in range(pretrained_config.data.class_count):
            # Compute and save the adversarial gradients.
            x_grad = get_saliency_map(model, x, j)
            save_image(x_grad, os.path.join(output_root, 'x_{}_grad_{}.png'.format(i, j)), normalize=True,
                       scale_each=True)
        break

    # Produce joint image.
    nrow = config.visualization.num_rows
    x_sub = to_cuda(torch.zeros(nrow, *x.size()[1:]).copy_(x[:nrow]).detach(), config.cuda)
    print("Size of visualization: ", x_sub.size(), "Maximum pixel value: ", x_sub.max())
    tensors = []
    c = 0
    for i, (x, y) in enumerate(data['test']):
        for (k, t) in enumerate(y):
            if t == c:
                c += 1
                tensors.append(x[k])
                if len(tensors) == pretrained_config.data.class_count:
                    break
        if len(tensors) == pretrained_config.data.class_count:
            break

    # Collect tensors from each class
    x_sub = to_cuda(torch.stack(tensors, 0), cuda=config.cuda)

    tensors = [x_sub]
    for j in range(pretrained_config.data.class_count):

        # Compute and visualize the adversarial gradients.
        model.zero_grad()
        x_grad = get_saliency_map(model, x_sub, j).clone().detach()
        tensors.append(x_grad)

    # Concatenate and visualize.
    joint_tensor = torch.cat(tensors, dim=0)
    save_image(joint_tensor, os.path.join(output_root, 'x_joint.png'), nrow=pretrained_config.data.class_count,
               normalize=True, colormap='seismic')
    # print("Train sigma(J): {}".format(check_grad_norm(model, data['train'], config.cuda)))
    # print("Val sigma(J): {}".format(check_grad_norm(model, data['validation'], config.cuda)))
    # print("Test sigma(J): {}".format(check_grad_norm(model, data['test'], config.cuda)))

if __name__ == '__main__':
    cfg = process_config()
    visualize_saliency(cfg)
