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
    model_name_in,
    vocoder_name_in = None):
    path = Path(__file__).parent / "../.models.json"
    manager = ModelManager(path)

    model_path, config_path, model_item = manager.download_model(model_name_in)
    vocoder_name = model_item["default_vocoder"] if vocoder_name_in is None else vocoder_name_in

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

    if model_name_in == 'tts_models/en/vctk/sc-glow-tts':
        wav = synthesizer.tts(speech_text, 'p244')
    else:
        wav = synthesizer.tts(speech_text)

    synthesizer.save_wav(wav, output_file_path)

if __name__ == '__main__':
    path = Path(__file__).parent / "data.json"
    with path.open() as f:
        # test = list(csv.reader(f))
        data = json.load(f)
    # f = open('data.json')

    list_of_models_to_test = [
        'tts_models/en/vctk/sc-glow-tts',
        'tts_models/en/ljspeech/tacotron2-DDC',
        'tts_models/en/ljspeech/tacotron2-DDC_ph',
        'tts_models/en/ljspeech/fast_pitch',
        'tts_models/en/ljspeech/speedy-speech',
    ]

    list_of_vocoders = [
        'vocoder_models/en/ljspeech/hifigan_v2'
    ]

    for item in data:
        txt = item['summarized_article'][0]['summary_text']
        uuid = item['uuid']
        for model_in_list in list_of_models_to_test:
            prefix = model_in_list.split("/")[-1]
            file_path = workspace_path + '/ttsoutput/' + prefix + '_' + uuid + '.wav'
            thisMakesAudio(txt, file_path, model_in_list)
