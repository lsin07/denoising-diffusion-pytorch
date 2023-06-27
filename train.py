import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from torchvision.utils import save_image

device = 'cuda' if torch.cuda.is_available else 'cpu'

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
).to(device)

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000,           # number of steps
).to(device)

trainer = Trainer(
    diffusion,
    './dataset/castle_128',
    train_batch_size = 32,
    train_lr = 8e-5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    calculate_fid = False             # whether to calculate fid during training
)

trainer.train()

# load pretrained model
# trainer.load(100)

training_images = torch.rand(8, 3, 128, 128).to(device) # images are normalized from 0 to 1
loss = diffusion(training_images)
loss.backward()

sampled_images = diffusion.sample(batch_size = 4)
save_image(sampled_images, 'result.png')
