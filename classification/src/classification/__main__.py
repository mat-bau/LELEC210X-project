import json
import os
import pickle
from pathlib import Path

import click
import requests

import common
from auth import PRINT_PREFIX
from common.env import load_dotenv
from common.logging import logger
from leaderboard.utils import get_url

from .utils import payload_to_melvecs

load_dotenv()

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def _load_model_and_voter(model_dir: str, classnames_fallback: list = None):
    """
    Load Keras model, config, classnames, extractor and RealTimeVoter
    from a model directory.

    Supports both config.json (new) and model_config.pkl (legacy).
    Returns (keras_model, voter, classnames) or (None, None, None).
    """
    if not model_dir or not os.path.exists(model_dir):
        return None, None, None

    try:
        import keras
        from classification.inference.predict import MelExtractor, RealTimeVoter, load_config

        cfg = load_config(model_dir)

        model_path = os.path.join(model_dir, "best_model.keras")
        if not os.path.exists(model_path):
            model_path = os.path.join(model_dir, "best_model.h5")
        keras_model = keras.models.load_model(model_path)

        meta_path = os.path.join(model_dir, "model_config.pkl")
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            classnames = meta["classnames"]
        elif classnames_fallback:
            classnames = classnames_fallback
        else:
            classnames = ["chainsaw", "fire", "fireworks", "gunshot"]

        extractor = MelExtractor(cfg)
        voter = RealTimeVoter(classnames=classnames, cfg=cfg)
        return keras_model, voter, classnames, extractor
    except Exception as e:
        logger.warning(f"Could not load ResNet model: {e}")
        return None, None, None, None


@click.command()
@click.option(
    "-i",
    "--input",
    "_input",
    default="-",
    type=click.File("r"),
    help="Where to read the input stream. Default to '-', a.k.a. stdin.",
)
@click.option(
    "-m",
    "--model",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the trained classification model (.keras or .h5).",
)
@click.option(
    "--model-dir",
    default=None,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory containing best_model.keras + config.json (preferred over --model).",
)
@common.click.melvec_length
@common.click.n_melvecs
@click.option(
    "--submit/--no-submit",
    default=True,
    help="Whether to submit the guesses to the leaderboard.",
)
@click.option(
    "-u",
    "--url",
    default=None,
    envvar="LEADERBOARD_URL",
    show_default=True,
    show_envvar=True,
    help="Base API url. If not specified, will use FLASK_RUN_HOST and FLASK_RUN_PORT.",
)
@click.option(
    "-k",
    "--key",
    default=None,
    envvar="LEADERBOARD_KEY",
    show_envvar=True,
    help="Your private key.",
)
@common.click.verbosity
def main(
    _input: click.File | None,
    model: Path | None,
    model_dir: Path | None,
    melvec_length: int,
    n_melvecs: int,
    submit: bool,
    url: str | None,
    key: str | None,
) -> None:
    """
    Extract Mel vectors from payloads and perform classification on them.

    If --model-dir is given, uses the new ResNet pipeline with RealTimeVoter
    (background filtering + majority voting). Otherwise falls back to legacy
    pickle model.

    Most likely, you want to pipe this script after running authentification:

        uv run auth | uv run classify --model-dir path/to/model_dir -k MY_KEY
    """
    if submit:
        if key is None:
            raise click.UsageError("You must provide a key to submit guesses.")
        url = url or get_url()

    # -- Try loading new ResNet pipeline -------------------------------------
    keras_model, voter, classnames, extractor = (None, None, None, None)
    if model_dir:
        keras_model, voter, classnames, extractor = _load_model_and_voter(
            str(model_dir))
        if keras_model:
            logger.info(f"ResNet model loaded. Classes: {classnames}")

    # -- Legacy pickle model fallback ----------------------------------------
    legacy_m = None
    if model and not keras_model:
        with open(model, "rb") as file:
            legacy_m = pickle.load(file)

    for payload in _input:
        if PRINT_PREFIX in payload:
            payload = payload[len(PRINT_PREFIX):]

            melvecs = payload_to_melvecs(payload, melvec_length, n_melvecs)
            logger.info(f"Parsed payload into Mel vectors: {melvecs}")

            guess = None

            if keras_model and voter and extractor:
                try:
                    import numpy as np
                    # melvecs: (n_melvecs, melvec_length) uint16 from firmware
                    # Reshape and feed to MelExtractor as a float spectrogram
                    spec = melvecs.astype(np.float32)
                    spec = (spec - spec.mean()) / (spec.std() + 1e-8)
                    # Resize to model input shape using scipy
                    import scipy.ndimage
                    n_mel_target = extractor.cfg.N_MEL
                    n_frames_target = extractor.n_frames
                    zoom = [n_mel_target / spec.shape[0],
                            n_frames_target / spec.shape[1]]
                    spec_resized = scipy.ndimage.zoom(spec, zoom, order=1)
                    tensor = spec_resized[np.newaxis, ..., np.newaxis].astype(np.float32)
                    probs = keras_model.predict(tensor, verbose=0)[0]
                    result = voter.update(probs)
                    if result is not None:
                        guess = result
                        logger.info(f"Voter decision: {guess}  probs={probs}")
                    else:
                        logger.debug("Voter: no submission this packet.")
                except Exception as e:
                    logger.error(f"ResNet inference error: {e}")

            elif legacy_m:
                # Legacy path — original TODO stub replaced with model call
                try:
                    import numpy as np
                    feat = melvecs.flatten().reshape(1, -1)
                    if hasattr(legacy_m, "predict"):
                        pred = legacy_m.predict(feat)
                        guess = str(pred[0])
                    else:
                        guess = "fire"
                except Exception as e:
                    logger.error(f"Legacy model error: {e}")
                    guess = "fire"

            if guess and submit:
                response = requests.post(
                    f"{url}/lelec210x/leaderboard/submit/{key}/{guess}"
                )
                response_as_dict = json.loads(response.text)
                if response.status_code == 200:
                    logger.info(response_as_dict)
                else:
                    logger.error(response_as_dict)
