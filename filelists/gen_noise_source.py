import os
import random
import pandas as pd
import soundfile as sound






train_clean_dir = ['/home/koredata/hsinhung/speech/vb_demand/clean_trainset_28spk_wav']
train_noisy_dir = ['/home/koredata/hsinhung/speech/vb_demand/noisy_trainset_28spk_wav']
test_clean_dir = ['/home/koredata/hsinhung/speech/vb_demand/clean_testset_wav']
test_noisy_dir = ['/home/koredata/hsinhung/speech/vb_demand/noisy_testset_wav']

train_noise_dir = '/home/koredata/hsinhung/speech/vb_demand/noise_trainset_28spk_wav'
test_noise_dir = '/home/koredata/hsinhung/speech/vb_demand/noise_testset_wav'
'''
for path in train_noisy_dir:
    for (dirpath, dirnames, filenames) in os.walk(path):
        print(dirpath)
        #print(len(filenames))
        for f in filenames:
            if not f.endswith((".WAV", ".wav")):
                continue
            
            clean_path = dirpath
            clean_path = clean_path.replace("noisy", "clean")
            
            noisy_path = os.path.join(dirpath, f)
            clean_path = os.path.join(clean_path, f)
            
            noisy, fs = sound.read(noisy_path)
            clean, fs = sound.read(clean_path)
            noise = noisy - clean
            noise_path = os.path.join(train_noise_dir, f)
            sound.write(noise_path, noise, fs)

          
print('==== train set ready ====')
'''

for path in test_noisy_dir:
    for (dirpath, dirnames, filenames) in os.walk(path):
        print(dirpath)
        #print(len(filenames))
        for f in filenames:
            if not f.endswith((".WAV", ".wav")):
                continue

            clean_path = dirpath
            clean_path = clean_path.replace("noisy", "clean")
            
            noisy_path = os.path.join(dirpath, f)
            clean_path = os.path.join(clean_path, f)
            
            noisy, fs = sound.read(noisy_path)
            clean, fs = sound.read(clean_path)
            noise = noisy - clean
            noise_path = os.path.join(test_noise_dir, f)
            sound.write(noise_path, noise, fs)


print('==== test set ready ====')

