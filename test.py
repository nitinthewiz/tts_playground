import torch
synthesizer = torch.hub.load('coqui-ai/TTS:v0.2.0', 
                             'tts', 
                             source='github')
wav = synthesizer.tts("This is an open-source library that generates synthetic speech!")
synthesizer.save_wav(wav, './ttsoutput/test_output.wav')