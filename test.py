import torch
synthesizer = torch.hub.load('coqui-ai/TTS:dev', 
                             'tts', 
                             source='github')
wav = synthesizer.tts("This is an open-source library that generates synthetic speech!")
synthesizer.save_wav(wav, './ttsoutput/test_output.wav')