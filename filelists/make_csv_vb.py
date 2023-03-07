import os
import random
import pandas as pd



f_train = open("train_vb.txt", "w")
f_val = open("val_vb.txt", "w")
f_test = open("test_vb.txt", "w")



train_clean_dir = ['/home/koredata/hsinhung/speech/vb_demand/clean_trainset_28spk_wav']
train_noisy_dir = ['/home/koredata/hsinhung/speech/vb_demand/noisy_trainset_28spk_wav']
test_clean_dir = ['/home/koredata/hsinhung/speech/vb_demand/clean_testset_wav']
test_noisy_dir = ['/home/koredata/hsinhung/speech/vb_demand/noisy_testset_wav']
train_noise_dir = '/home/koredata/hsinhung/speech/vb_demand/noise_trainset_28spk_wav'
test_noise_dir = '/home/koredata/hsinhung/speech/vb_demand/noise_testset_wav'

for path in train_noisy_dir:
    for (dirpath, dirnames, filenames) in os.walk(path):
        print(dirpath)
        #print(len(filenames))
        for f in filenames:
            if not f.endswith((".WAV", ".wav")):
                continue
            
            clean_path = dirpath
            clean_path = clean_path.replace("noisy", "clean")
            noise_path = dirpath.replace("noisy", "noise")
            
            noisy_path = os.path.join(dirpath, f)
            clean_path = os.path.join(clean_path, f)
            noise_path = os.path.join(noise_path, f)
            
            coin = random.random()
            if coin <= 0.2:
                f_val.write(noisy_path)
                f_val.write(' ')
                f_val.write(clean_path)
                f_val.write(' ')
                f_val.write(noise_path)
                f_val.write('\n')
            elif coin <= 1:
                f_train.write(noisy_path)
                f_train.write(' ')
                f_train.write(clean_path)
                f_train.write(' ')
                f_train.write(noise_path)
                f_train.write('\n')
          
print('==== train set ready ====')

for path in test_noisy_dir:
    for (dirpath, dirnames, filenames) in os.walk(path):
        print(dirpath)
        #print(len(filenames))
        for f in filenames:
            if not f.endswith((".WAV", ".wav")):
                continue

            clean_path = dirpath
            clean_path = clean_path.replace("noisy", "clean")
            noise_path = dirpath.replace("noisy", "noise")
            
            noisy_path = os.path.join(dirpath, f)
            clean_path = os.path.join(clean_path, f)
            noise_path = os.path.join(noise_path, f)

            f_test.write(noisy_path)
            f_test.write(' ')
            f_test.write(clean_path)
            f_test.write(' ')
            f_test.write(noise_path)
            f_test.write('\n')
          



f_train.close()
f_val.close()
f_test.close()

