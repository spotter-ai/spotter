from gym.envs.registration import register

register(id='CupsWorld-v0',
         entry_point='gym_cupsworld.envs:CupsWorldEnv',
         )