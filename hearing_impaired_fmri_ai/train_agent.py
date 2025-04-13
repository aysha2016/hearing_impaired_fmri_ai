import dreamerv3
from dreamerv3 import embodied
from audio_game_simulation import AudioGameEnv

config = embodied.Config(dreamerv3.configs['defaults'])
config = config.update({'logdir': './log/audio_game', 'run.train_ratio': 64})
config = embodied.Flags(config).parse()

env = AudioGameEnv()
dreamerv3.train(env, config)