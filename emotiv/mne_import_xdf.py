import gzip
import logging
import struct
import xml.etree.ElementTree as ET
from glob import glob
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from pyxdf import load_xdf

logger = logging.getLogger()


def open_xdf(filename):
    """Open XDF file for reading."""
    filename = Path(filename)  # convert to pathlib object
    if filename.suffix == '.xdfz' or filename.suffixes == ['.xdf', '.gz']:
        f = gzip.open(filename, 'rb')
    else:
        f = open(filename, 'rb')
    if f.read(4) != b'XDF:':  # magic bytes
        raise IOError('Invalid XDF file {}'.format(filename))
    return f


def match_streaminfos(stream_infos, parameters):
    """Find stream IDs matching specified criteria.

    Parameters
    ----------
    stream_infos : list of dicts
        List of dicts containing information on each stream. This information
        can be obtained using the function resolve_streams.
    parameters : list of dicts
        List of dicts containing key/values that should be present in streams.
        Examples: [{"name": "Keyboard"}] matches all streams with a "name"
                  field equal to "Keyboard".
                  [{"name": "Keyboard"}, {"type": "EEG"}] matches all streams
                  with a "name" field equal to "Keyboard" and all streams with
                  a "type" field equal to "EEG".
    """
    matches = []
    for request in parameters:
        for info in stream_infos:
            for key in request.keys():
                match = info[key] == request[key]
                if not match:
                    break
            if match:
                matches.append(info['stream_id'])

    return list(set(matches))  # return unique values


def resolve_streams(fname):
    """Resolve streams in given XDF file.

    Parameters
    ----------
    fname : str
        Name of the XDF file.

    Returns
    -------
    stream_infos : list of dicts
        List of dicts containing information on each stream.
    """
    return parse_chunks(parse_xdf(fname))


def parse_xdf(fname):
    """Parse and return chunks contained in an XDF file.

    Parameters
    ----------
    fname : str
        Name of the XDF file.

    Returns
    -------
    chunks : list
        List of all chunks contained in the XDF file.
    """
    chunks = []
    with open_xdf(fname) as f:
        for chunk in _read_chunks(f):
            chunks.append(chunk)
    return chunks


def _read_chunks(f):
    """Read and yield XDF chunks.

    Parameters
    ----------
    f : file handle
        File handle of XDF file.


    Yields
    ------
    chunk : dict
        XDF chunk.
    """
    while True:
        chunk = dict()
        try:
            chunk["nbytes"] = _read_varlen_int(f)
        except EOFError:
            return
        chunk["tag"] = struct.unpack('<H', f.read(2))[0]
        if chunk["tag"] in [2, 3, 4, 6]:
            chunk["stream_id"] = struct.unpack("<I", f.read(4))[0]
            if chunk["tag"] == 2:  # parse StreamHeader chunk
                xml = ET.fromstring(f.read(chunk["nbytes"] - 6).decode())
                chunk = {**chunk, **_parse_streamheader(xml)}
            else:  # skip remaining chunk contents
                f.seek(chunk["nbytes"] - 6, 1)
        else:
            f.seek(chunk["nbytes"] - 2, 1)  # skip remaining chunk contents
        yield chunk


def _parse_streamheader(xml):
    """Parse stream header XML."""
    return {el.tag: el.text for el in xml if el.tag != "desc"}


def parse_chunks(chunks):
    """Parse chunks and extract information on individual streams."""
    streams = []
    for chunk in chunks:
        if chunk["tag"] == 2:  # stream header chunk
            streams.append(dict(stream_id=chunk["stream_id"],
                                name=chunk.get("name"),  # optional
                                type=chunk.get("type"),  # optional
                                source_id=chunk.get("source_id"),  # optional
                                created_at=chunk.get("created_at"),  # optional
                                uid=chunk.get("uid"),  # optional
                                session_id=chunk.get("session_id"),  # optional
                                hostname=chunk.get("hostname"),  # optional
                                channel_count=int(chunk["channel_count"]),
                                channel_format=chunk["channel_format"],
                                nominal_srate=int(chunk["nominal_srate"])))
    return streams


def read_raw_xdf(fname, stream_id=None):
    """Read XDF file.

    Parameters
    ----------
    fname : str
        Name of the XDF file.
    stream_id : int | str | None
        ID (number) or name of the stream to load (optional). If None, the
        first stream of type "EEG" will be read.

    Returns
    -------
    raw : mne.io.Raw
        XDF file data.
    """
    streams, header = load_xdf(fname)

    if stream_id is not None:
        if isinstance(stream_id, str):
            stream = _find_stream_by_name(streams, stream_id)
        elif isinstance(stream_id, int):
            stream = _find_stream_by_id(streams, stream_id)
    else:
        stream = _find_stream_by_type(streams, stream_type="EEG")

    if stream is not None:
        name = stream["info"]["name"][0]
        n_chans = int(stream["info"]["channel_count"][0])
        fs = float(stream["info"]["nominal_srate"][0])
        logger.info(f"Found EEG stream '{name}' ({n_chans} channels, "
                    f"sampling rate {fs}Hz).")
        print(f"Found EEG stream '{name}' ({n_chans} channels, "
              f"sampling rate {fs}Hz).")

        labels, types, units = _get_ch_info(stream)  # itt halt meg a szentem:c

        if not labels:
            labels = [str(n) for n in range(n_chans)]
        if not units:
            units = ["NA" for _ in range(n_chans)]
        info = mne.create_info(ch_names=labels, sfreq=fs, ch_types="eeg")
        # convert from microvolts to volts if necessary
        scale = np.array([1e-6 if u == "microvolts" else 1 for u in units])
        raw = mne.io.RawArray((stream["time_series"] * scale).T, info)
        first_samp = stream["time_stamps"][0]
    else:
        logger.info("No EEG stream found.")
        return

    markers = _find_stream_by_type(streams, stream_type="Markers")
    if markers is not None:
        onsets = markers["time_stamps"] - first_samp
        logger.info(f"Adding {len(onsets)} annotations.")
        descriptions = markers["time_series"]

        # átalakítani a descriptionst 1D-vé, mert value errort dobott előtte

        # descriptions = list(np.concatenate(descriptions).flat)

        annotations = mne.Annotations(onsets, [0] * len(onsets), np.squeeze(np.array(descriptions)))
        raw.set_annotations(annotations)

    return raw


def _find_stream_by_name(streams, stream_name):
    """Find the first stream that matches the given name."""
    for stream in streams:
        if stream["info"]["name"][0] == stream_name:
            return stream


def _find_stream_by_id(streams, stream_id):
    """Find the stream that matches the given ID."""
    for stream in streams:
        if stream["info"]["stream_id"] == stream_id:
            return stream


def _find_stream_by_type(streams, stream_type="EEG"):
    """Find the first stream that matches the given type."""
    for stream in streams:
        if stream["info"]["type"][0] == stream_type:
            return stream


def _get_ch_info(stream):
    labels, types, units = [], [], []
    if stream["info"]["desc"]:
        print("_get_ch_info: ")
        for ch in stream["info"]["desc"][0]["channels"][0]["channel"]:
            labels.append(str(ch["label"][0]))
            types.append(ch["type"][0])
            units.append(ch["unit"][0])
    return labels, types, units


def _read_varlen_int(f):
    """Read a variable-length integer."""
    nbytes = f.read(1)
    if nbytes == b'\x01':
        return ord(f.read(1))
    elif nbytes == b'\x04':
        return struct.unpack('<I', f.read(4))[0]
    elif nbytes == b'\x08':
        return struct.unpack('<Q', f.read(8))[0]
    elif not nbytes:  # EOF
        raise EOFError
    else:
        raise RuntimeError('Invalid variable-length integer encountered.')


if __name__ == "__main__":

    # fnames = glob("C:/Users/User/Documents/Rita/1etem/5.szemeszter/CurrentStudy/sub-P001/ses-S001/eeg/sub-P001_ses-S001_task-Default_run-001_eeg.xdf")
    # fnames = glob("C:/Users/User/Documents/Rita/1etem/5.szemeszter/CurrentStudy/Keyboard only/sub-P001/ses-S001/eeg/sub-P001_ses-S001_task-Default_run-001_eeg.xdf")
    # fnames = glob("C:/Users/User/Documents/Rita/1etem/5.szemeszter/CurrentStudy/EEG_w_Keyboard/sub-P001/ses-S001/eeg/sub-P001_ses-S001_task-Default_run-001_eeg.xdf")
    # fnames = glob("C:/Users/User/Documents/Rita/1etem/5.szemeszter/BCI FOT/xdf_sample.xdf")

    fnames = glob(
        "C:/Users/User/Documents/Rita/1etem/Database/emotiv/paradigmC/sub-P001_run-001_eeg.xdf")
    for fname in fnames:
        print("=" * len(fname) + "\n" + fname + "\n" + "=" * len(fname))
        raw = read_raw_xdf(fname)
        if raw is not None:
            print("raw is not None")
            print(raw, end="\n\n")
            print(raw.annotations, end="\n\n")
            print(raw.ch_names)
            # data = raw.get_data()
            # time = raw.times
            raw.filter(l_freq=30.0, h_freq=None)
            # raw.crop(tmin=150)
            # raw.plot_psd(area_mode='range', tmax=10.0, show=False, average=True)
            raw.plot(n_channels=14, show_first_samp=True, block=True)  # pip install matplotlib==3.2.0, higher versions don't work

        chunks = parse_xdf(fname)

        df = pd.DataFrame.from_dict(chunks)
        df = df[["nbytes", "tag", "stream_id"]]
        df["stream_id"] = df["stream_id"].astype("Int64")
        df["tag"] = pd.Categorical(df["tag"], ordered=True)
        df["tag"].cat.rename_categories(["FileHeader", "StreamHeader", "Samples",
                                         "ClockOffset", "Boundary",
                                         "StreamFooter"], inplace=True)

        print("Chunk table\n-----------")
        print(df, end="\n\n")  # detailed chunk table

        print("Chunk type frequencies\n----------------------")
        print(df["tag"].value_counts().sort_index(), end="\n\n")

        print("Chunks per stream\n-----------------")
        print(df["stream_id"].value_counts().sort_index(), end="\n\n")

        print("Unique stream IDs\n-----------------")
        print(sorted(df["stream_id"].dropna().unique()), end="\n\n")
