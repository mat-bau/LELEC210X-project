import sys, pickle
sys.path.insert(0, 'classification/src')
from classification.configs import BaseConfig

with open("/Users/marcbauvir/Matéo/Cours/LELEC210X-project/data/models/matéo_mcu_match_20260414_1527/model_config.pkl", "rb") as f:
    meta = pickle.load(f)

cfg = BaseConfig(
    N_MEL=meta["n_mel"],
    N_FFT=meta["n_fft"],
    HOP_LENGTH=meta["hop_length"],
    SAMPLE_RATE=meta["sample_rate"],
    MODEL_DIR="data/models/models_resnet/lunS8_10_50",
)
cfg.to_json("/Users/marcbauvir/Matéo/Cours/LELEC210X-project/data/models/matéo_mcu_match_20260414_1527/config.json")
print("config.json créé")