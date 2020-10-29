import os
import sys
import tempfile


def read_audios(data_dir, file_ext=['wav', 'mp3', 'sph'], file_per_dir=sys.maxsize):
    for r, ds, fs in os.walk(data_dir):
        for f in fs[:file_per_dir]:
            if f.split('.')[-1] in file_ext:
                if f.endswith('.sph'):
                    from sphfile import SPHFile;
                    sph = SPHFile(os.path.join(r, f))
                    with tempfile.NamedTemporaryFile(prefix=f) as tempf:
                        sph.write_wav(tempf.name)
                        yield os.path.dirname(tempf.name), os.path.basename(tempf.name)
                else:
                    yield r, f


# quick testing
if __name__ == '__main__':
    files = read_audios(sys.argv[1])
    while True:
        try:
            print(next(files))
        except StopIteration:
            break
