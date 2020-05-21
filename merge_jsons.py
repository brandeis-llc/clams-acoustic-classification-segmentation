import json
import os
import sys
import json2txt


def merge_jsonfiles(json_fnames, out_fname):
    json.dump(merge_synced_dicts(adjust_all(json_fnames)), open(out_fname, 'w'))


def adjust_all(json_fnames):
    for fname in json_fnames:
        """
        (keigh fork) exp/tri3/decode-cpb-aacip-259-dj58gh9t.h264.mp4.11.79.wav_16kHz.wav
        (hipstas v1) exp/tri3/decode-cpb-aacip-259-dj58gh9t.h264.mp4.11.79_16kHz.wav
        """
        fbase = os.path.basename(fname)
        offset = float('.'.join(fbase.split('_16kHz')[0].split('.')[-2:]))
        yield adjust(fname, offset)


def merge_synced_dicts(adjusteds):
    words = [] 
    for adjusted in adjusteds:
        words.extend(adjusted['words'])
    words = sorted(words, key=lambda x: x['time'])
    return {'words': words}


def adjust(json_fname, offset):
    with open(json_fname) as in_f:
        synced = json.loads(in_f.read(), encoding='utf-8')
        for word in synced['words']:
            word['time'] = round(offset + word['time'], 2)
    return synced
            

if __name__ == "__main__":
    d = sys.argv[1]
    if d.endswith('/'):
        d = d[:-1]
    dparent, dbase= os.path.split(d)
    split_json_dir = os.path.join(d, 'transcripts', 'json')
    merge_json_dir = os.path.join(dparent, 'transcripts', 'json')
    merge_txt_dir = os.path.join(dparent, 'transcripts', 'txt')
    merge_json_fname = os.path.join(merge_json_dir, dbase + '.json')
    merge_txt_fname = os.path.join(merge_txt_dir, dbase + '.txt')
    os.makedirs(merge_json_dir, exist_ok=True)
    os.makedirs(merge_txt_dir, exist_ok=True)
    merge_jsonfiles([os.path.join(split_json_dir, f) for f in os.listdir(split_json_dir)], merge_json_fname)
    json2txt.convert(merge_json_fname, merge_txt_fname)


