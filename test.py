# import torch
# synthesizer = torch.hub.load('coqui-ai/TTS:v0.2.0', 
#                              'tts', 
#                              source='github')
# wav = synthesizer.tts("This is an open-source library that generates synthetic speech!")
# synthesizer.save_wav(wav, './ttsoutput/test_output.wav')

import os
from pathlib import Path
from TTS.utils.synthesizer import Synthesizer
from TTS.utils.manage import ModelManager

model_path = None
config_path = None
speakers_file_path = None
vocoder_path = None
vocoder_config_path = None
encoder_path = None
encoder_config_path = None
use_cuda = False

model_name = 'tts_models/en/ljspeech/tacotron2-DDC'
vocoder_name = 'vocoder_models/en/ljspeech/hifigan_v2'

path = Path(__file__).parent / "../.models.json"
manager = ModelManager(path)

model_path, config_path, model_item = manager.download_model(model_name)
vocoder_path, vocoder_config_path, _ = manager.download_model(vocoder_name)

synthesizer = Synthesizer(
    model_path,
    config_path,
    speakers_file_path,
    vocoder_path,
    vocoder_config_path,
    encoder_path,
    encoder_config_path,
    use_cuda,
)

wav = synthesizer.tts("This is an open-source library that generates synthetic speech!")
workspace_path = os.environ['GITHUB_WORKSPACE']
synthesizer.save_wav(wav, workspace_path + '/ttsoutput/test_output.wav')
