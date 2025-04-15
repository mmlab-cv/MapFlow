import os
import torch.optim as optim
from absl import app
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.mapflow import *
from src.utils import *

from config.settings import FLAGS


def train(FLAGS, train, val, model, optimizer):
    for epoch in range(FLAGS.epochs):
        model.train()
        model.training = True
        with tqdm(train,unit="trn_batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            epoch_loss = 0
            mean_g_prob = 0
            num_samples = 0
            for batch_idx, (x, y_true) in enumerate(tepoch):
                x = x.to(FLAGS.device)
                y_true = y_true.to(FLAGS.device)

                prob, g_prob, y_enc, x_t, angle = model.log_prob(y_true, x)

                if FLAGS.ae_enabled:
                    y_dec = model.autoencoder.decoding(y_enc)
                    y_abs = model._rel_to_abs(y_dec, x_t)
                    y = model._rotate(y_abs, x_t, -1*angle, inference=True)

                loss = -prob.mean()

                gaussian_prob = torch.exp(g_prob)/(1/torch.sqrt(torch.tensor(2*torch.pi)))
                mean_g_prob += gaussian_prob.mean().item()
                epoch_loss += -prob.sum().item()
                num_samples += x.size(0)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if FLAGS.wandb:
                    wandb.log({"train/loss": (epoch_loss/num_samples)}, step=epoch)

                tepoch.set_postfix(loss=(epoch_loss/num_samples))

            if (epoch+1) % FLAGS.validation_rate == 0:
                check(FLAGS=FLAGS, dataset=val, model=model, z=None, epoch=(epoch+1))

    save_checkpoints(FLAGS, model)
    return epoch




def test(FLAGS, test, model):
    model.eval
    
    ade_tot, fde_tot, min_ade, min_fde = check(FLAGS, test, model)
    monotonic_ade = 0
    monotonic_fde = 0
    for i in range(len(ade_tot)-1):
        if ade_tot[i+1]-ade_tot[i] >= 0:
            monotonic_ade += 1
        if fde_tot[i+1]-fde_tot[i] >= 0:
            monotonic_fde += 1
    
    if FLAGS.wandb:
        wandb.log({"test/monotonic_ade": monotonic_ade})
        wandb.log({"test/monotonic_fde": monotonic_fde})
        
    print(f"Min ade: {min_ade}")
    print(f"Min fde: {min_fde}")
    print(f"Sequential ade: {monotonic_ade}")
    print(f"Sequential fde: {monotonic_fde}")




def main(argv):
    # 1. Init experiment
    set_seed(FLAGS.seed)
    if FLAGS.wandb:
        wandb.init(config=FLAGS, project="MapFlow")

    # 2. Get Data
    trainset, valset, testset = get_data(FLAGS)

    # 3. Init network and optimizer
    model = MapFlow(FLAGS)
    
    if FLAGS.load_checkpoints:
        model.load_state_dict(torch.load(FLAGS.weights_file)["model"])

    model.autoencoder.load_state_dict(torch.load("weights/autoencoder/AugAE16316.pth"))
    for param in model.autoencoder.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr)

    # 4. Train or test
    if FLAGS.train:
        epoch = train(FLAGS, trainset, valset, model, optimizer)
    if not(FLAGS.train) or epoch == (FLAGS.epochs-1):
        test(FLAGS, testset, model)

if __name__=='__main__':
    app.run(main)