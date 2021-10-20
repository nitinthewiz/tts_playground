# import torch
# synthesizer = torch.hub.load('coqui-ai/TTS:v0.2.0', 
#                              'tts', 
#                              source='github')
# wav = synthesizer.tts("This is an open-source library that generates synthetic speech!")
# synthesizer.save_wav(wav, './ttsoutput/test_output.wav')


from TTS.utils.synthesizer import Synthesizer
from TTS.utils.manage import ModelManager

model_name = 'tts_models/en/ljspeech/tacotron2-DDC'
vocoder_name = 'vocoder_models/en/ljspeech/hifigan_v2'

synthesizer = Synthesizer(
    model_path,
    config_path,
    speakers_file_path,
    vocoder_path,
    vocoder_config_path,
    encoder_path,
    encoder_config_path,
    args.use_cuda,
)