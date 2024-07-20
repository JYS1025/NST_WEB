import torch, torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from PIL import Image
from torchvision.models import VGG19_Weights, vgg19


if torch.cuda.is_available():
      device = torch.device("cuda")
      print("CUDA is available. Using GPU.")
else:
      device = torch.device("cpu")
      print("CUDA is not available. Using CPU.")


def gram_matrix(target_layer_outputs, generate_layer_outputs):
    gram_target_list = []
    gram_generate_list = []

    for target_layer, generate_layer in zip(target_layer_outputs, generate_layer_outputs):
        batch_size, channels, height, width = generate_layer.shape
        target_layer = target_layer.view(channels, -1)
        generate_layer = generate_layer.view(channels, -1)

        gram_target = torch.mm(target_layer, target_layer.t())
        gram_generate = torch.mm(generate_layer, generate_layer.t())

        gram_target_list.append(gram_target)
        gram_generate_list.append(gram_generate)

    return gram_target_list, gram_generate_list

def get_content_loss(content_layer_outputs, generate_layer_outputs):
    criterion = nn.MSELoss()
    if torch.cuda.is_available():
        criterion = criterion.to(device)
    total_content_loss = 0
    for content_output, generate_output in zip(content_layer_outputs, generate_layer_outputs):
        content_loss = criterion(content_output, generate_output)
        total_content_loss += content_loss
    return total_content_loss/len(content_layer_outputs)

def get_style_loss(gram_target_list, gram_generate_list):
    criterion = nn.MSELoss()
    if torch.cuda.is_available():
        criterion = criterion.to(device)
    total_style_loss = 0
    N=0
    for gram_target, gram_generate in zip(gram_target_list, gram_generate_list):
        style_loss = criterion(gram_target, gram_generate)
        N = gram_target.shape[0]
        style_loss = style_loss/(4*(N**2)*81)
        total_style_loss += style_loss
    return total_style_loss/len(gram_target_list)




def main(content_img_path, target_img_path, output_img_path):

    epochs = 2000

    alpha = 1.0
    beta = 10.0

    content_img = Image.open(content_img_path)
    target_img = Image.open(target_img_path)

    generated_img = None

    #import pre-trained vgg net
    vgg_content = torchvision.models.vgg19(VGG19_Weights.DEFAULT)
    vgg_target = torchvision.models.vgg19(VGG19_Weights.DEFAULT)
    vgg_generate = torchvision.models.vgg19(VGG19_Weights.DEFAULT)

    if torch.cuda.is_available():
        vgg_content = vgg_content.to(device)
        vgg_target = vgg_target.to(device)
        vgg_generate = vgg_generate.to(device)
    #preprocess for input images
    size = content_img.size
    content_img = content_img.resize(size)
    target_img = target_img.resize(size)

    weights = VGG19_Weights.DEFAULT
    preprocess = weights.transforms()

    content_img = preprocess(content_img).unsqueeze(0)  # Add batch dimension
    target_img = preprocess(target_img).unsqueeze(0)  # Add batch dimension

    #initialize generated image to noised image
    # generated_img = generated_img = content_img.clone()
    generated_img = torch.randn(content_img.size())
    if torch.cuda.is_available():
        content_img = content_img.to(device)
        target_img = target_img.to(device)
        generated_img = generated_img.to(device)
    generated_img.requires_grad_(True)
    #-----------track each vgg layers output-----------#
    content_layer_outputs = []
    target_layer_outputs = []
    generate_layer_outputs = []

    def get_layer_output_hook_content(module, input, output):
        if isinstance(module, nn.Conv2d):  # Check if the module is a convolutional layer
            content_layer_outputs.append(output)
    def get_layer_output_hook_target(module, input, output):
        if isinstance(module, nn.Conv2d):
            target_layer_outputs.append(output)
    def get_layer_output_hook_generate(module, input, output):
        if isinstance(module, nn.Conv2d):
            generate_layer_outputs.append(output)

    hooks_content = []
    for name, layer in vgg_content.named_modules():
        hooks_content.append(layer.register_forward_hook(get_layer_output_hook_content))

    hooks_target = []
    for name, layer in vgg_target.named_modules():
        hooks_target.append(layer.register_forward_hook(get_layer_output_hook_target))

    hooks_generate = []
    for name, layer in vgg_generate.named_modules():
        hooks_generate.append(layer.register_forward_hook(get_layer_output_hook_generate))
    #--------------------------------------------------#

    #Define loss fn and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.LBFGS([generated_img], lr=0.07)

    #Define mean and std for undo-normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    for i in range(epochs):
        #initialize each epoch
        optimizer.zero_grad()
        content_layer_outputs = []
        target_layer_outputs = []
        generate_layer_outputs = []

        _ = vgg_content(content_img)
        _ = vgg_target(target_img)
        _ = vgg_generate(generated_img)

        #content loss
        content_loss = get_content_loss(content_layer_outputs, generate_layer_outputs)
        # print(content_loss.shape)

        # style_loss
        gram_target, gram_generate = gram_matrix(target_layer_outputs, generate_layer_outputs)
        style_loss = get_style_loss(gram_target, gram_generate)
        # print(style_loss.shape)

        #total_style_loss
        loss_total = alpha * content_loss + beta * style_loss

        loss_total.backward()
        optimizer.step(closure=lambda: loss_total)
        
        if(i%100 == 0):
            visualize_img = generated_img.clone().detach().cpu().squeeze(0)
            # 역정규화 적용
            visualize_img = visualize_img * std + mean
            # 텐서 변환 및 채널 순서 변경
            visualize_img = visualize_img.permute(1, 2, 0).numpy()
            # 픽셀 값 범위 조정 및 데이터 타입 변환
            visualize_img = (visualize_img * 255).clip(0, 255).astype(np.uint8)
            plt.imshow(visualize_img)
            plt.show()
            print(f"Epoch {i+1}/{epochs}, Content Loss: {content_loss.item()}, Style Loss: {style_loss.item()}")

    for hook in hooks_content:
        hook.remove()
    for hook in hooks_target:
        hook.remove()
    for hook in hooks_generate:
        hook.remove()

    #undo-normalization on generated img
    generated_img = generated_img.detach().cpu().squeeze(0)
    generated_img = generated_img * std + mean
    generated_img = generated_img.permute(1, 2, 0).numpy()
    generated_img = (generated_img * 255).clip(0, 255).astype(np.uint8)

    # Save the generated image
    Image.fromarray(generated_img).save(output_img_path)

    return

if __name__ == "__main__":
    main()