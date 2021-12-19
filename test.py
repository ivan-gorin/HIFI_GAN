from src.parse_config import ConfigParser
import torch
import torchaudio
import argparse
import os
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence


@torch.no_grad()
def main(config: ConfigParser, resume_path: str):
    logger = config.get_logger('Test')
    model = config.get_model()
    featurizer = config.get_melspectrogram()
    device = config.device
    logger.info("Loading checkpoint: {} ...".format(resume_path))
    checkpoint = torch.load(resume_path, device)

    # load architecture params from checkpoint.
    model.load_state_dict(checkpoint["state_dict"])
    logger.info("Checkpoint Loaded.")
    model.eval()

    test_audio = []
    testdir = Path(config['data']['test']['dir'])
    names = os.listdir(testdir)
    for file in names:
        audio, sr = torchaudio.load(testdir / file)
        test_audio.append(audio.squeeze(0))

    waveform = pad_sequence(test_audio).transpose(0, 1).to(device)
    specs = featurizer(waveform)
    logger.info("Synthesizing audio...")
    pred_wav = model(specs).cpu()
    logger.info("Audio synthesized. Saving...")
    sr = config['melspectrogram']['sample_rate']
    for i in range(len(pred_wav)):
        path = config.save_dir / f'Synthesized_{names[i]}.wav'
        torchaudio.save(path, pred_wav[i], sr)
    logger.info("Saved in {}".format(config.save_dir))


if __name__ == '__main__':
    args = argparse.ArgumentParser('test')
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
        required=True
    )
    args.add_argument(
        "-m",
        "--model",
        default=None,
        type=str,
        help="model checkpoint file path (default: None)",
        required=True
    )
    parsed = args.parse_args()
    configparser = ConfigParser(parsed.config)
    main(configparser, parsed.model)
