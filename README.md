# Audio Segmenter

Simple audio segmenter to isolate speech portion of audio files. Uses a simple feedforward MLP for classification and heuristic smoothing methods to increase speech recall. 


## Requirements 

* System packages: `ffmpeg`
* Python packages: 
  * `librosa`
  * `tensorflow` or `tensorflow-gpu`
  * `numpy`
  * `scipy`
  * `scikit-learn`

## Training 

### Pretrained model 

We provide a [pretrained model](pretrained/). The model is trained on [MUSAN corpus](https://www.openslr.org/17/), using binary labels (`speech` vs. `nonspeech`). The model is serialized using [`tensorflow::SavedModel` format](https://www.tensorflow.org/guide/keras/save_and_serialize#export_to_savedmodel). Because of the distribution bias in the corpus (a lot of speech recordings in the data), we randomly resampled from speech portion to match the size to negative examples. In doing so, the language distribution among the resampled speech examples was NOT deliberately balanced. 

### Training pipeline

To train your own model, invoke `run.py` with `-t` flag and pass the directory name where training data is stored. You might also want to take a look at `extract_all` function in `feature.py` to change how the labels are read in. 

## Segmentation

To run the segmenter over audio files, invoke `run.py` with `-s` flag, and pass model path (feel free to use the pretrained model if needed) and the directory where audio files are stored. Currently it will process all `mp3` and `wav` files in the target directory. If you want to add other types of audio file, add or change the `file_ext` list at the bottom of `run.py` files. 

The processed results are stored as `segmented.tsv`, a tab-separated file, in the target directory. Each row of the file represents a single file, and columns represents; 
* first column shows the file path
* last column shows the ratio of speech portion if the file 
* columns between are paired into start and end points (in seconds) of speech segments. 

### Using docker

We also provide [`Dockerfile`](Dockerfile). If you want to run the segmenter as a docker container (not worrying about dependencies), build an image from the `Dockerfile` and run it with the target directory mounted to `/segmenter/data`. Just MAKE SURE that target directory is writable by others (`chmod u+w $TARGET_DIR`). For example, 

```bash
chmod -R u+w $HOME/audio-files && docker build . -t audioseg && docker run --rm -v $HOME/audio-files:/segmenter/data -it audioseg
```

Once the process is done, you'll find a `segmented.tsv` file in the local target directory. 
