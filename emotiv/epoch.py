import numpy as np
from pandas import DataFrame
from scipy.io import loadmat
from pyedflib import EdfWriter, FILETYPE_EDFPLUS

from preprocess import open_raw_file

WANTED_CHANNELS = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8',
                   'AF4']  # , 'GYROX', 'GYROY', 'MARKER']
TRIGGER_LIST = 'trigger_list'
FILE_APPEND = '_trigger'


def add_trigger_to_edf_file(edf_file, trigger_file):
    """This function adds triggers to an edf file from a matlab cell array.

    The edf file should contain markers > 0, which are indicates time points where the
    triggers should be added.

    Parameters
    ----------
    edf_file: str
        EDF file recorded from EmotivXavierTestBench
    trigger_file: str
        MATLAB .mat file, which contains all the trigger names. The parameter name in
        the .mat file should be equivalent with the TRIGGER_LIST variable.

    Returns
    -------

    """
    raw = open_raw_file(edf_file)
    fs = raw.info['sfreq']
    channel_info = [
        {'label': ch, 'dimension': 'V', 'sample_rate': fs, 'physical_max': 0.01, 'physical_min': 0.0,
         'digital_max': 2 ** 14, 'digital_min': 0, 'transducer': '', 'prefilter': ''} for ch in
        raw.info['ch_names'] if ch in WANTED_CHANNELS]

    eeg = raw.get_data()
    df = DataFrame(eeg.T, columns=raw.info['ch_names'])
    # df = raw.to_data_frame()
    df2 = df[df.MARKER > 0].MARKER  # triggers
    trigger_inds = np.array(df2.index)

    raw.pick_channels(WANTED_CHANNELS)
    eeg = raw.get_data()
    n_ch = raw.info['nchan']
    raw.close()
    del raw

    # trg_file = select_eeg_file_in_explorer(message="Select trigger file", file_type='MATLAB', ext='.mat')
    mat = loadmat(trigger_file)
    triggers = mat[TRIGGER_LIST]
    triggers = [triggers[0, r][0] for r in range(len(triggers[0]))]

    ext = '.' + edf_file.split('.')[-1]
    ext_ind = edf_file.find(ext)
    out_file = edf_file[:ext_ind] + FILE_APPEND + edf_file[ext_ind:]

    edf = EdfWriter(out_file, n_ch, file_type=FILETYPE_EDFPLUS)
    edf.setSignalHeaders(channel_info)
    edf.writeSamples(eeg)
    for i, ind in enumerate(trigger_inds):
        edf.writeAnnotation(ind / fs, -1, triggers[i])

    edf.close()
    del edf


if __name__ == '__main__':
    from gui_handler import select_file_in_explorer

    edf_file = select_file_in_explorer(message='Select EDF file!', file_type='Emotiv Epoc EDF', ext='.edf')
    trigger_file = select_file_in_explorer(message='Select trigger .mat file!', file_type='MATLAB', ext='.mat')
    add_trigger_to_edf_file(edf_file, trigger_file)
