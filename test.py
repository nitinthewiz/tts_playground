import os
import json
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

workspace_path = os.environ['GITHUB_WORKSPACE']


def thisMakesAudio(
    speech_text,
    output_file_path,
    model_name = 'tts_models/en/ljspeech/tacotron2-DDC',
    vocoder_name = 'vocoder_models/en/ljspeech/hifigan_v2'):
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
    synthesizer.save_wav(wav, workspace_path + '/ttsoutput/test_output.wav')

if __name__ == '__main__':
    path = Path(__file__).parent / "data.json"
    with path.open() as f:
        # test = list(csv.reader(f))
        data = json.load(f)
    # f = open('data.json')
    for item in data:
        txt = item['summarized_article'][0]['summary_text']
        uuid = item['uuid']
        file_path = workspace_path + '/ttsoutput/' + uuid + '.wav'
    thisMakesAudio(txt, file_path)