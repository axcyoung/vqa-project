#!pip install byol-pytorch

import torch
from byol_pytorch import BYOL
#from torchvision import models

#assume variable name of model instance is sparse_graph_model

learner = BYOL(
    sparse_graph_model,
    image_size = 256, #image dimension
    hidden_layer = 'avgpool', #name/index of hidden layer, whose output is used as the latent representation
    #projection_size = 256,           # the projection size
    #projection_hidden_size = 4096,   # the hidden dimension of the MLP for both the projection and prediction
    #moving_average_decay = 0.99      # the moving average decay factor for the target encoder
)

opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

def sample_unlabelled_images():
    return torch.randn(20, 3, 256, 256)

for _ in range(100):
    images = sample_unlabelled_images()
    loss = learner(images)
    opt.zero_grad()
    loss.backward()
    opt.step()
    learner.update_moving_average() # update moving average of target encoder

# save your improved network
torch.save(resnet.state_dict(), './improved-net.pt')