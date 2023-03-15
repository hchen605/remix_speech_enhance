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
        "source_1_pesq": [], 
        "source_1_stoi": [],
        "source_2_pesq": [], 
        "source_2_stoi": []
    }
    
    ### check whether the inference file has specific sample rate. Our default sample rate is 16000
    if "sample_rate" in inference_args:
        sample_rate = inference_args["sample_rate"]
    else:
        sample_rate = 16000

    for noisy, clean_1, clean_2, name in tqdm(dataloader, desc="Inference"):
        assert len(name) == 1, "The batch size of inference stage must 1."
        name = name[0]

        if noisy.size(1) < 16000*0.5:
            print(f"Warning! {name} is too short for computing STOI. Will skip this for now.")
            continue

        noisy = noisy.to(device)
        enhanced = model(noisy)

        noisy = noisy.squeeze().cpu().numpy()
        source_1 = enhanced[:,0,:]
        source_2 = enhanced[:,1,:]
        source_1 = source_1.squeeze().cpu().numpy()
        source_2 = source_2.squeeze().cpu().numpy()
        clean_1 = clean_1.squeeze().numpy()
        clean_2 = clean_2.squeeze().numpy()

        noisy_stoi = STOI(clean_1, noisy, sr=sample_rate)
        source_1_stoi = STOI(clean_1, source_1, sr=sample_rate)

        if (noisy_stoi == 1e-5) or (source_1_stoi == 1e-5):
            assert source_1_stoi == noisy_stoi
            print(f" {name} skip the length check.")
            continue
        
        try:
            result["noisy_pesq"].append(PESQ(clean_1, noisy, sr=sample_rate))
            result["source_1_pesq"].append(PESQ(clean_1, source_1, sr=sample_rate))
            result["source_2_pesq"].append(PESQ(clean_1, source_2, sr=sample_rate))
        except NoUtterancesError:
            print("can't found utterence in {}! ignore it".format(name))
            continue
        except ValueError:
            result["noisy_pesq"].append(1)
            result["source_1_pesq"].append(1)
            result["source_2_pesq"].append(1)


        result["filename"].append(name)
        result["noisy_stoi"].append(STOI(clean_1, noisy, sr=sample_rate))
        result["source_1_stoi"].append(STOI(clean_1, source_1, sr=sample_rate))
        result["source_2_stoi"].append(STOI(clean_1, source_2, sr=sample_rate))


        sf.write(enhanced_dir / f"{name}_source_1.wav", source_1, samplerate=sample_rate)
        sf.write(enhanced_dir / f"{name}_source_2.wav", source_2, samplerate=sample_rate)
        #sf.write(enhanced_dir / f"{name}_before.wav", noisy, samplerate=sample_rate)
        #sf.write(enhanced_dir / f"{name}_clean.wav", clean, samplerate=sample_rate)


    df = pd.DataFrame(result)
    print("NOISY PESQ: {:.4f} ± {:.4f}".format(*mean_std(df["noisy_pesq"].to_numpy())))
    print("NOISY STOI: {:.4f} ± {:.4f}".format(*mean_std(df["noisy_stoi"].to_numpy())))
    print("Source 1 PESQ: {:.4f} ± {:.4f}".format(*mean_std(df["source_1_pesq"].to_numpy())))
    print("Source 1 STOI: {:.4f} ± {:.4f}".format(*mean_std(df["source_1_stoi"].to_numpy())))
    print("Source 2 PESQ: {:.4f} ± {:.4f}".format(*mean_std(df["source_2_pesq"].to_numpy())))
    print("Source 2 STOI: {:.4f} ± {:.4f}".format(*mean_std(df["source_2_stoi"].to_numpy())))
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
