FROM tensorflow/tensorflow:2.4.0

RUN apt-get update && apt-get install -y libsndfile1 ffmpeg

RUN useradd -d /segmenter -m segmenter  && chown -R segmenter /segmenter
COPY . /segmenter
WORKDIR /segmenter
RUN pip install -r requirements.txt
RUN python setup.py develop
USER segmenter

CMD bacs -s /segmenter/pretrained/$(ls pretrained/ | sort | tail -1) /segmenter/data > /segmenter/data/segmented.tsv
