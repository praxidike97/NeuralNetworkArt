import torch
from torchvision import models, transforms
from torch import optim
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
style_weights = {'conv1_1': 1.,
                 'conv2_1': 0.8,
                 'conv3_1': 0.5,
                 'conv4_1': 0.3,
                 'conv5_1': 0.1}
content_weight = 1
style_weight = 1e5
checkpoint = 500


def load_image(image_path, size=500):
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        # Normalize image, see https://pytorch.org/docs/stable/torchvision/models.html
        transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], std=[0.225, 0.224, 0.229]),
    ])
    image = transform(image).unsqueeze(0).to(device)
    return image


def gram_matrix(x):
    size_batch, channels, height, width = x.shape
    # Flatten the tensor
    x = x.view(channels, height*width)
    # Multiply matrix with transposed matrix
    return torch.mm(x, torch.t(x))


def get_feature_output(image, model, layers=None):
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '21': 'conv4_2',  ## content representation
                  '28': 'conv5_1'
                 }
    feature_output = {}
    for name, layer in model._modules.items():
        image = layer(image)
        if name in layers:
            feature_output[layers[name]] = image
    return feature_output


image_content = load_image("./content_image.png")
image_style = load_image("./style_image02.jpeg")

#plt.imshow(image.numpy()[0].transpose((1, 2, 0)))
#plt.show()

# Selecting features because we don't have any use for the classifier
vgg = models.vgg19(pretrained=True).features
vgg.to(device)
# The weights of the model will not be changes! Just the combined image
for parameters in vgg.parameters():
    parameters.requires_grad_(False)


# Get the feature maps
content_features = get_feature_output(image_content, vgg)
style_features = get_feature_output(image_style, vgg)

# Calculate gram matrices
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# The combined image is the content image at first, then we apply backpropagation
image_combined = image_content.clone().requires_grad_(True).to(device)

# Optimizer
optimizer = optim.Adam([image_combined], lr=0.001)

epochs = 4000

for i in tqdm(range(epochs)):

    target_features = get_feature_output(image_combined, vgg)

    # Content loss
    content_loss = torch.mean((target_features["conv4_2"] - content_features["conv4_2"]) ** 2)

    # Clearing  the accumilated gradients
    optimizer.zero_grad()

    # Computing style loss
    style_loss = 0
    for layer in style_weights:
        target_gram = gram_matrix(target_features[layer])
        layer_loss = style_weights[layer] * torch.mean((target_gram - style_grams[layer]) ** 2)
        b, d, h, w = image_combined.shape
        style_loss += layer_loss / (d * h * w)

    total_loss = content_weight * content_loss + style_weight * style_loss

    # Backpropagation
    total_loss.backward()
    optimizer.step()

    # Displaying target image at checkpoints
    if (i + 1) % checkpoint == 0:
        print('Total loss: ', total_loss.item())
        print(image_combined.to("cpu").clone().detach().squeeze().numpy().shape)
        numpy_image = image_combined.to("cpu").clone().detach().squeeze().numpy().transpose((1, 2, 0))* np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
        numpy_image = numpy_image.clip(0, 1)
        #plt.imshow(numpy_image)
        print(numpy_image.shape)
        #output_image = Image.fromarray(numpy_image)
        #output_image = Image.fromarray(np.uint8(numpy_image * 255))
        #output_image.save("./output-{}.png".format(str(i+1).zfill(3)))
        plt.imsave("./output-{}.png".format(str(i+1).zfill(3)), numpy_image)
        #plt.show()
