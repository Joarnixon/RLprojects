import torch
from dqn import Agent

def load_checkpoint(filepath, grad=False):
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    model = Agent(3)
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = grad

    model.eval()

    return model

def save_checkpoint(model):
    checkpoint = {'model': Agent(3), 'state_dict': model.state_dict()}

    torch.save(checkpoint, 'checkpoint.pth')
    print('Saved')