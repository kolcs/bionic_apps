from BCISystem import BCISystem, Features, Databases

CMD_IN = 1.6  # sec

if __name__ == '__main__':
    print('Starting BCI System for game play...')
    inp = input('Select paradigm for pilots (C -- 4 way, D -- binary):    C / D\n')
    if inp.upper() == 'C':
        db_name = Databases.GAME_PAR_C
    elif inp.upper() == 'D':
        db_name = Databases.GAME_PAR_D
    else:
        raise NotImplementedError('Can not run BCI System with paradigm {}'.format(inp))

    bci = BCISystem(feature=Features.FFT_RANGE)
    bci.play_game(db_name=db_name,
                  fft_low=2, fft_high=40,
                  window_length=1,
                  window_step=0.1,
                  epoch_tmax=4,
                  command_in_each_sec=CMD_IN)
