FROM tensorflow/tensorflow:2.0.1

RUN apt-get update && apt-get install -y libsndfile1 ffmpeg

RUN useradd -d /segmenter -m segmenter  && chown -R segmenter /segmenter
USER segmenter
COPY . /segmenter
WORKDIR /segmenter
RUN pip install -r requirements.txt

CMD python run.py -s pretrained/$(ls pretrained/ | sort | tail -1) data > data/segmented.tsv
