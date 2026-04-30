import sys, pickle
sys.path.insert(0, 'classification/src')
from classification.configs import BaseConfig

with open("data/models/models_resnet/lunS8_10_50/model_config.pkl", "rb") as f:
    meta = pickle.load(f)

cfg = BaseConfig(
    N_MEL=meta["n_mel"],
    N_FFT=meta["n_fft"],
    HOP_LENGTH=meta["hop_length"],
    SAMPLE_RATE=meta["sample_rate"],
    MODEL_DIR="data/models/models_resnet/lunS8_10_50",
)
cfg.to_json("data/models/models_resnet/lunS8_10_50/config.json")
print("config.json créé")