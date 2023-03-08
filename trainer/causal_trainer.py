import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from trainer.base_trainer import BaseTrainer
from utility.sdr import *

plt.switch_backend('agg')


class Trainer(BaseTrainer):
    def __init__(self, config, resume: bool, model, optimizer, scheduler, train_dataloader, validation_dataloader):
        super(Trainer, self).__init__(config, resume, model, optimizer, scheduler)
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader

    def _train_epoch(self, epoch):
        loss_total = 0.0

        pbar = tqdm(self.train_dataloader)
        for noisy, clean_1, clean_2, name in pbar:
            self.optimizer.zero_grad()

            noisy = noisy.to(self.device)  # [Batch, length]
            clean_1 = clean_1.to(self.device)  # [Batch, length]
            clean_2 = clean_2.to(self.device)  # [Batch, length]

            clean = torch.unsqueeze(clean_1, 1) # [Batch, 1, length]
            clean = torch.stack((clean_1, clean_2),dim=1)

            #loss = self.model(noisy, clean_1)
            enhanced = self.model(noisy)
            sisnr = batch_SDR_torch(enhanced, clean)
            loss = -torch.mean(sisnr)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            loss_total += loss.item()
            pbar.set_description("Loss: {:.3f}".format(loss.item()))

        self.writer.add_scalar(f"Loss/Train", loss_total / len(self.train_dataloader), epoch)

    @torch.no_grad()
    def _validation_epoch(self, epoch):
        noisy_list = []
        clean_list = []
        enhanced_list = []

        loss_total = 0.0

        visualization_limit = self.validation_custom_config["visualization_limit"]


        for i, (noisy, clean_1, clean_2, name) in tqdm(enumerate(self.validation_dataloader), desc="Inference"):
            assert len(name) == 1, "The batch size of inference stage must 1."
            name = name[0]

            if noisy.size(1) < 16000*0.6:
                print(f"Warning! {name} is too short for computing STOI. Will skip this for now.")
                continue

            noisy = noisy.to(self.device)  # [Batch, length]
            clean_1 = clean_1.to(self.device)  # [Batch, length]
            clean_2 = clean_2.to(self.device)  # [Batch, length]

            clean = torch.unsqueeze(clean_1, 1) # [Batch, 1, length]
            clean = torch.stack((clean_1, clean_2),dim=1)

            loss = self.model(noisy, clean)
            #enhanced = self.model.inference(noisy)
            enhanced = self.model(noisy)
            sisnr = batch_SDR_torch(enhanced, clean)
            loss = -torch.mean(sisnr)

            loss_total += loss.item()
            noisy = noisy.squeeze(0).cpu().numpy()
            enhanced = enhanced[:,0,:]
            enhanced = enhanced.squeeze(0).cpu().numpy() # remove the batch dimension
            clean_1 = clean_1.squeeze(0).cpu().numpy()

            assert len(noisy) == len(clean_1) == len(enhanced)

            if i <= np.min([visualization_limit, len(self.validation_dataloader)]):
                self.spec_audio_visualization(noisy, enhanced, clean_1, name, epoch)

            noisy_list.append(noisy)
            clean_list.append(clean_1)
            enhanced_list.append(enhanced)

        self.writer.add_scalar(f"Loss/Validation", loss_total / len(self.validation_dataloader), epoch)
        return self.metrics_visualization(noisy_list, clean_list, enhanced_list, epoch)
