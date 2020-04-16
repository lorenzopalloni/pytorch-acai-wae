import torch
import torchvision

from torch.utils.tensorboard import SummaryWriter

from models import Encoder, Decoder
from preprocessing import get_loader, inv_standardize
from utils import (
    Collector,
    reconstruction_loss_func,
    wasserstein_penalty_func
)
from config import (
    knobs,
    log_dir_local_time,
    log_dir_last_modified,
    checkpoints_dir_local_time,
    checkpoints_dir_last_modified,
    interpolations_dir
)

loader = get_loader()

encoder = Encoder().to(knobs["device"])
decoder = Decoder().to(knobs["device"])

opt_encoder = torch.optim.Adam(encoder.parameters(), lr=knobs["lr_encoder"])
opt_decoder = torch.optim.Adam(decoder.parameters(), lr=knobs["lr_decoder"])

collector_reconstruction_loss = Collector()
collector_wasserstein_penalty = Collector()
collector_fooling_term = Collector()
collector_codes_min = Collector()
collector_codes_max = Collector()
if knobs["resume"]:
    writer = SummaryWriter(log_dir_last_modified)
    checkpoint_dir = checkpoints_dir_last_modified
    checkpoint = torch.load(checkpoint_dir)
    starting_epoch = checkpoint["epoch"]
    iteration = checkpoint["iteration"]
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    opt_encoder.load_state_dict(checkpoint["opt_encoder_state_dict"])
    opt_decoder.load_state_dict(checkpoint["opt_decoder_state_dict"])
else:
    writer = SummaryWriter(log_dir_local_time)
    checkpoint_dir = checkpoints_dir_local_time
    starting_epoch = 1
    iteration = 0

encoder.train()
decoder.train()
for epoch in range(starting_epoch, knobs["num_epochs"] + 1):
    for batch in loader:
        iteration += 1

        codes = encoder(batch)
        reconstructions = decoder(codes)

        codes_fake = (
                torch.randn(knobs["batch_size"], knobs["hidden_dim"]) * knobs["sigma"]
        ).to(knobs["device"])

        reconstruction_loss = reconstruction_loss_func(batch, reconstructions)
        wasserstein_penalty = wasserstein_penalty_func(codes, codes_fake)
        loss_autoencoder = (
                knobs["lambda_reconstruction"] * reconstruction_loss
                + knobs["lambda_penalty"] * wasserstein_penalty
        )

        opt_encoder.zero_grad()
        opt_decoder.zero_grad()
        loss_autoencoder.backward(retain_graph=True)
        if knobs["clip_gradient"]:
            torch.nn.utils.clip_grad_norm_(
                encoder.parameters(), knobs["max_norm_encoder"]
            )
            torch.nn.utils.clip_grad_norm_(
                decoder.parameters(), knobs["max_norm_decoder"]
            )
        opt_encoder.step()
        opt_decoder.step()

        with torch.no_grad():
            collector_reconstruction_loss.append(reconstruction_loss.cpu().numpy())
            collector_wasserstein_penalty.append(wasserstein_penalty.cpu().numpy())
            collector_codes_min.append(codes.min().cpu().numpy())
            collector_codes_max.append(codes.max().cpu().numpy())
        if iteration % knobs["time_to_collect"] == 0:
            encoder.eval()
            decoder.eval()
            print(
                "Epoch: {:5d} || ".format(epoch)
                + "#{:6d} || ".format(iteration)
                + "reconstruction_loss:\t{:0.6f} || ".format(
                    collector_reconstruction_loss.mean()
                )
                + "wasserstein_penalty:\t{:0.6f} || ".format(
                    collector_wasserstein_penalty.mean()
                )
            )
            writer.add_scalar(
                "Reconstruction_Loss_average_20_obs",
                collector_reconstruction_loss.mean(),
                iteration,
            )
            writer.add_scalar(
                "wasserstein_penalty_average_20_obs",
                collector_wasserstein_penalty.mean(),
                iteration
            )
            writer.add_scalar(
                "codes_min_over_20_obs",
                collector_codes_min.min(),
                iteration
            )
            writer.add_scalar(
                "codes_max_over_20_obs",
                collector_codes_max.max(),
                iteration
            )

            if iteration % (knobs["time_to_collect"] * 4) == 0:

                it_encoder_parameters = encoder.parameters()
                for k, v in encoder.state_dict().items():
                    if k.find("bias") != -1 or k.find("weight") != -1:
                        writer.add_histogram(
                            "encoder/" + k.replace(".", "/"), v, iteration
                        )
                        writer.add_histogram(
                            "encoder/" + k.replace(".", "/") + "/grad",
                            next(it_encoder_parameters).grad,
                            iteration,
                            )
                it_decoder_parameters = decoder.parameters()
                for k, v in decoder.state_dict().items():
                    if k.find("bias") != -1 or k.find("weight") != -1:
                        writer.add_histogram(
                            "decoder/" + k.replace(".", "/"), v, iteration
                        )
                        writer.add_histogram(
                            "decoder/" + k.replace(".", "/") + "/grad",
                            next(it_decoder_parameters).grad,
                            iteration,
                            )

                num_images = 16
                codes = encoder(batch[:num_images])
                num_cols = 8
                num_rows = num_images // 2
                interpolations = []
                for row in range(num_rows):
                    for col, level in enumerate(
                            torch.linspace(0, 1, num_cols).to(knobs["device"])
                    ):
                        interpolations.append(
                            inv_standardize(
                                decoder(
                                    torch.lerp(codes[2 * row], codes[2 * row + 1], level)
                                )
                                .detach()
                                .cpu()
                            )
                        )
                torchvision.utils.save_image(
                    torch.stack(interpolations, dim=1).squeeze(0),
                    fp=interpolations_dir / f'iter_{iteration}.jpg',
                    nrow=num_rows
                )

            torch.save(
                {
                    "epoch": epoch,
                    "iteration": iteration,
                    "encoder_state_dict": encoder.state_dict(),
                    "decoder_state_dict": decoder.state_dict(),
                    "opt_encoder_state_dict": opt_encoder.state_dict(),
                    "opt_decoder_state_dict": opt_decoder.state_dict(),
                },
                checkpoint_dir,
            )
            encoder.train()
            decoder.train()
writer.close()
