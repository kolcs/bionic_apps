import os
import sys
import urllib.request
import re

BASE_URL = "https://chmura.put.poznan.pl/s/G285gnQVuCnfQAx/download?path=%2F"

VIDEO_1080p_DIR = "Video-1080p"
VIDEO_576p_DIR = "Video-576p"
DEPTH_DIR = "Depth"
DATA_HDF5_DIR = "Data-HDF5"
DATA_CSV_DIR = "Data-CSV"


def usage():
    print("Usage: {:s} <experiment_type> <media_type> [<id1> <id2> ...]".format(os.path.basename(__file__)))
    print()
    print("Arguments:")
    print("    <experiment_type>    comma-separated list of experiment types "
          "(supported types: emg_gestures, emg_force)")
    print("    <media_type>         comma-separated list of media "
          "(supported types: data-csv, data-hdf5, depth, video-1080p, video-576p)")
    print("    [<id1> <id2> ...]    optional list of two-digit participant IDs, fetches all if none are given")
    print()
    print("Examples:")
    print("{:s} emg_gestures data-hdf5,video-1080p".format(os.path.basename(__file__)))
    print("{:s} emg_gestures,emg_force data-csv,depth 03 04 07".format(os.path.basename(__file__)))
    exit(1)


def parse_record(name):
    experiment_name_regexp = r"^(?P<type>\w*)-(?P<id>\d{2})-(?P<trajectory>\w*)-" \
                             r"(?P<date>\d{4}-\d{2}-\d{2})-(?P<time>\d{2}-\d{2}-\d{2}-\d{3})"
    tags = re.search(experiment_name_regexp, name)
    if not tags:
        raise Warning("Wrong record", name)
    return tags.group("type"), tags.group("id"), tags.group("trajectory"), tags.group("date"), tags.group("time")


def download_progress(blocknum, blocksize, totalsize):
    readsofar = blocknum * blocksize
    if totalsize > 0:
        percent = readsofar * 1e2 / totalsize
        s = "\r%5.1f%% %*d / %d" % (
            percent, len(str(totalsize)), readsofar, totalsize)
        sys.stderr.write(s)
        if readsofar >= totalsize:  # near the end
            sys.stderr.write("\n")
    else:  # total size is unknown
        sys.stderr.write("read %d\n" % (readsofar,))


def main():
    if len(sys.argv) < 3:
        print("Illegal number of parameters")
        usage()

    experiment_types = sys.argv[1].split(",")
    for e in experiment_types:
        if not e in ("emg_gestures", "emg_force"):
            print("Invalid experiment type \"{:s}\"".format(e))
            usage()
    experiment_types = set(experiment_types)

    media_types = sys.argv[2].split(",")
    for m in media_types:
        if not any(m in t for t in ("data-csv", "data-hdf5", "depth", "video-1080p", "video-576p")):
            print("Invalid media type \"{:s}\"".format(m))
            usage()
    media_types = set(media_types)

    print(experiment_types)
    print(media_types)

    records_available = urllib.request.urlopen(BASE_URL + "&files=records.txt").read().decode("utf-8").splitlines()

    records = list()
    ids = set()
    for r in records_available:
        experiment_type, id, trajectory, date, time = parse_record(r)
        records.append((experiment_type, id, trajectory, date, time))
        ids.add(id)

    ids_requested = set()
    if len(sys.argv) > 3:
        for id in sys.argv[3:]:
            if not re.match(r"^[0-9]{2}$", id):
                print("Invalid id \"{:s}\"".format(id))
                usage()
            if not id in ids:
                print("ID \"{:s}\" not available".format(id))
                exit(1)
            ids_requested.add(id)
        ids = ids.intersection(ids_requested)

    ids = list(ids)
    ids.sort()

    print(ids)

    if "data-csv" in media_types:
        os.makedirs(DATA_CSV_DIR, exist_ok=True)
    if "data-hdf5" in media_types:
        os.makedirs(DATA_HDF5_DIR, exist_ok=True)
    if "depth" in media_types:
        os.makedirs(DEPTH_DIR, exist_ok=True)
    if "video-1080p" in media_types:
        os.makedirs(VIDEO_1080p_DIR, exist_ok=True)
    if "video-576p" in media_types:
        os.makedirs(VIDEO_576p_DIR, exist_ok=True)

    for r in records:
        if r[0] in experiment_types:
            if r[1] in ids:
                record = "{:s}-{:s}-{:s}-{:s}-{:s}".format(r[0], r[1], r[2], r[3], r[4])
                if "data-csv" in media_types:
                    print(DATA_CSV_DIR + "/" + record + ".zip")
                    urllib.request.urlretrieve(BASE_URL + DATA_CSV_DIR + "&files=" + record + ".zip",
                                               DATA_CSV_DIR + "/" + record + ".zip", download_progress)
                if "data-hdf5" in media_types:
                    print(DATA_HDF5_DIR + "/" + record + ".hdf5")
                    urllib.request.urlretrieve(BASE_URL + DATA_HDF5_DIR + "&files=" + record + ".hdf5",
                                               DATA_HDF5_DIR + "/" + record + ".hdf5", download_progress)
                if "depth" in media_types and r[0] != "emg_force":
                    print(DEPTH_DIR + "/" + record + ".zip")
                    urllib.request.urlretrieve(BASE_URL + DEPTH_DIR + "&files=" + record + ".zip",
                                               DEPTH_DIR + "/" + record + ".zip", download_progress)
                if "video-1080p" in media_types:
                    print(VIDEO_1080p_DIR + "/" + record + ".mp4")
                    urllib.request.urlretrieve(BASE_URL + VIDEO_1080p_DIR + "&files=" + record + ".mp4",
                                               VIDEO_1080p_DIR + "/" + record + ".mp4", download_progress)
                if "video-576p" in media_types:
                    print(VIDEO_576p_DIR + "/" + record + ".mp4")
                    urllib.request.urlretrieve(BASE_URL + VIDEO_576p_DIR + "&files=" + record + ".mp4",
                                               VIDEO_576p_DIR + "/" + record + ".mp4", download_progress)


if __name__ == '__main__':
    main()
