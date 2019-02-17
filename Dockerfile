FROM alpine:3.8 as parser_trainer

# we need the real GNU patch (the busybox version is not sufficient!)
RUN apk update && apk add git python2 py2-gflags py2-pip patch && \
    pip install nltk

WORKDIR /opt
RUN git clone https://github.com/kaayy/josydipa.git


# Copy the relevant files of the RST-DT and PTB corpora, so we can merge them.

ADD rst_discourse_treebank/data/RSTtrees-WSJ-main-1.0/TRAINING/*.out.dis /opt/josydipa/dataset/rst/train/
ADD rst_discourse_treebank/data/RSTtrees-WSJ-main-1.0/TEST/*.out.dis /opt/josydipa/dataset/rst/test/
ADD pennTreebank/parsed/mrg/wsj/*/*.mrg /opt/josydipa/dataset/ptb/

# There are five files in RST-DT whose name doesn't match their PTB counterpart.
# For more information, see the RST-DT README.
ADD rst_discourse_treebank/data/RSTtrees-WSJ-main-1.0/TRAINING/file1.dis /opt/josydipa/dataset/rst/train/wsj_0764.out.dis
ADD rst_discourse_treebank/data/RSTtrees-WSJ-main-1.0/TRAINING/file2.dis /opt/josydipa/dataset/rst/train/wsj_0430.out.dis
ADD rst_discourse_treebank/data/RSTtrees-WSJ-main-1.0/TRAINING/file3.dis /opt/josydipa/dataset/rst/train/wsj_0766.out.dis
ADD rst_discourse_treebank/data/RSTtrees-WSJ-main-1.0/TRAINING/file4.dis /opt/josydipa/dataset/rst/train/wsj_0778.out.dis
ADD rst_discourse_treebank/data/RSTtrees-WSJ-main-1.0/TRAINING/file5.dis /opt/josydipa/dataset/rst/train/wsj_2172.out.dis


# Apply patches to the RST Discourse Treebank files and Penn Treebank 
# files. This step is necessary because there are some small mismatches 
# between the RST Discourse tree texts and the Penn tree texts.
ADD mrg2cleangold.py /opt/josydipa/

RUN apk add python3 py3-pip parallel && \
    pip3 install nltk==3.4

WORKDIR /opt/josydipa/dataset/rst/train
RUN patch -p0 < ../../../patches/rst-ptb.train.patch

WORKDIR /opt/josydipa/dataset/rst/test
RUN patch -p0 < ../../../patches/rst-ptb.test.patch

WORKDIR /opt/josydipa/dataset/ptb
RUN ls *.mrg | parallel ../../mrg2cleangold.py {} {.}.cleangold
RUN patch -p0 < ../../patches/ptb-rst.patch


WORKDIR /opt/josydipa
RUN python src/tokenize_rst.py --rst_path dataset/rst/train && \
    python src/tokenize_rst.py --rst_path dataset/rst/test

RUN mkdir -p dataset/joint && \
    python src/aligner.py --rst_path dataset/rst/train --const_path dataset/ptb > dataset/joint/train.txt && \
    python src/aligner.py --rst_path dataset/rst/test --const_path dataset/ptb > dataset/joint/test.txt


