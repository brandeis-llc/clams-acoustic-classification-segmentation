# Acoustic Calssification & Segmentation 

Simple audio segmenter to isolate speech portion out of audio streams. Uses a simple feedforward neural network for classification (implemented using `tensorflow`) and heuristic smoothing methods to increase the recall of speech segments. 


## Requirements and installation

* System packages: [`ffmpeg`](http://ffmpeg.org/download.html)
* Installation: install `brandeis-acs` from PyPI 

    ```
    pip install brandeis-acs
    ```

## Training 

### Pretrained model 

We provide a [pretrained model](pretrained/). The model is trained on [MUSAN corpus](https://www.openslr.org/17/), using binary labels (`speech` vs. `nonspeech`). The model is, then, serialized using [`tensorflow::SavedModel` format](https://www.tensorflow.org/guide/keras/save_and_serialize#export_to_savedmodel). Because of the distribution bias in the corpus (a lot more of speech recordings in the training data), we randomly resampled from frames (size of 10ms) from speech examples to match its size to negative examples. In doing so, the language distribution among the resampled speech examples was NOT deliberately balanced. 

### Training pipeline

To train your own model, invoke `bacs` with `-t` flag and pass the directory name where training data is stored. You might also want to take a look at `extract_all` function in [`feature.py`](bacs/feature.py) to change how the labels are read in, if using corpora other than the MUSAN. 

## Segmentation

To run the segmenter over audio files, invoke `bacs` with `-s` flag, and pass the directory where audio files are stored. Optionally, you can pass the model path with `-m` flag. If the model path is not given, the default pretrained model will be used. Currently, it will process all `mp3` and `wav` files in the target directory. 

If you want to process other types of audio file, you need to edit source code for now. Clone this repository and add to or change the `file_ext` list near the bottom of [`run.py`](bacs/run.py) files. When running from source code, run `run.py` file. 

The processed results are stored as `segmented.tsv`, a tab-separated file, in the target directory. Each row of the file represents a result from a single audio file, and columns represents as follows; 
* first column shows the file path
* last column shows the ratio of speech portion of the file 
* columns between are paired into start and end points (in seconds) of speech segments. 

### Using docker

We also provide [`Dockerfile`](Dockerfile). If you want to run the segmenter as a docker container (not worrying about dependencies), first build an image from this project directory using the `Dockerfile`. Note that the image will not use the PyPI package version, but copy the code as of the build-time. 
Then run the image with the target directory with audio files to process mounted at `/segmenter/data`. Just MAKE SURE that target directory is writable by others (`chmod o+w $TARGET_DIR`) because a non-root user will be running the processor in the container. For example, 

```bash
git clone https://github.com/keighrim/audio-segmentation.git 
cd audio-segmentation
chmod -R o+w $HOME/audio-files && docker build . -t audioseg && docker run --rm -v $HOME/audio-files:/segmenter/data -it audioseg
```

Once the process is done, you'll find a `segmented.tsv` file in the local target directory. 
