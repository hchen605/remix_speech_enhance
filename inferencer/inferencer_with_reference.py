import os
from numpy import short

import torch
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from inferencer.base_inferencer import BaseInferencer
from util.metrics import STOI, PESQ, mean_std

from pesq import NoUtterancesError


@torch.no_grad()
def inference_wrapper(dataloader, model, device, inference_args, enhanced_dir):

    result = {
        "filename": [], 
        "noisy_pesq": [], 
        "noisy_stoi": [],
        "enhanced_pesq": [], 
        "enhanced_stoi": [],
    }
    
    ### check whether the inference file has specific sample rate. Our default sample rate is 16000
    if "sample_rate" in inference_args:
        sample_rate = inference_args["sample_rate"]
    else:
        sample_rate = 16000

    for noisy, clean, name in tqdm(dataloader, desc="Inference"):
        assert len(name) == 1, "The batch size of inference stage must 1."
        name = name[0]

        if noisy.size(1) < 16000*0.5:
            print(f"Warning! {name} is too short for computing STOI. Will skip this for now.")
            continue

        noisy = noisy.to(device)
        enhanced = model.inference(noisy)

        noisy = noisy.squeeze().cpu().numpy()
        enhanced = enhanced.squeeze().cpu().numpy()
        clean = clean.squeeze().numpy()

        noisy_stoi = STOI(clean, noisy, sr=sample_rate)
        enhanced_stoi = STOI(clean, enhanced, sr=sample_rate)

        if (noisy_stoi == 1e-5) or (enhanced_stoi == 1e-5):
            assert enhanced_stoi == noisy_stoi
            print(f" {name} skip the length check.")
            continue
        
        try:
            result["noisy_pesq"].append(PESQ(clean, noisy, sr=sample_rate))
            result["enhanced_pesq"].append(PESQ(clean, enhanced, sr=sample_rate))
        except NoUtterancesError:
            print("can't found utterence in {}! ignore it".format(name))
            continue
        except ValueError:
            result["noisy_pesq"].append(1)
            result["enhanced_pesq"].append(1)


        result["filename"].append(name)
        result["noisy_stoi"].append(STOI(clean, noisy, sr=sample_rate))
        result["enhanced_stoi"].append(STOI(clean, enhanced, sr=sample_rate))


        sf.write(enhanced_dir / f"{name}_after.wav", enhanced, samplerate=sample_rate)
        #sf.write(enhanced_dir / f"{name}_before.wav", noisy, samplerate=sample_rate)
        #sf.write(enhanced_dir / f"{name}_clean.wav", clean, samplerate=sample_rate)


    df = pd.DataFrame(result)
    print("NOISY PESQ: {:.4f} ± {:.4f}".format(*mean_std(df["noisy_pesq"].to_numpy())))
    print("NOISY STOI: {:.4f} ± {:.4f}".format(*mean_std(df["noisy_stoi"].to_numpy())))
    print("ENHANCED PESQ: {:.4f} ± {:.4f}".format(*mean_std(df["enhanced_pesq"].to_numpy())))
    print("ENHANCED STOI: {:.4f} ± {:.4f}".format(*mean_std(df["enhanced_stoi"].to_numpy())))
    df.to_csv(os.path.join(enhanced_dir, "_results.csv"), index=False)


class Inferencer(BaseInferencer):
    def __init__(self, config, checkpoint_path, output_dir):
        super(Inferencer, self).__init__(config, checkpoint_path, output_dir)

    @torch.no_grad()
    def inference(self):
        inference_wrapper(
            dataloader=self.dataloader,
            model=self.model,
            device=self.device,
            inference_args=self.inference_config,
            enhanced_dir=self.enhanced_dir,
        )
