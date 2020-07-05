import librosa
import librosa.display
import matplotlib.pyplot as plt
import pathlib

data_path = pathlib.Path('E:\\6_dB_fan\\id_02\\normal')
all_image_paths = list(data_path.glob('*'))
# print(len(all_image_paths))
# print(all_image_paths[0:10]) # 1016,2049,
all_image_paths = [str(path) for path in all_image_paths]

path2 = 'E:\\6db2png'
nums = 0
for path in all_image_paths[400:464]:
    y, sr = librosa.load(path, sr=None)
    melspec = librosa.feature.melspectrogram(y, sr, n_fft=1024, hop_length=512, n_mels=128)
    logmelspec = librosa.power_to_db(melspec)
    librosa.display.specshow(logmelspec)
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.savefig(path2 + '\\' + str(nums)+'.png')
    nums = nums + 1
