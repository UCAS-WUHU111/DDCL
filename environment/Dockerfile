# hash:sha256:e97af9bf49a7a886eee81509ac11dd1401a9a255bbf4e1dd79b6ae955a0850f7
FROM registry.codeocean.com/codeocean/miniconda3:4.9.2-cuda11.7.0-cudnn8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y --channel=pytorch \
        cudatoolkit==11.3.1 \
        h5py==3.7.0 \
        python==3.9.15 \
        pytorch==1.11.0 \
        tensorflow==2.5.0 \
        tqdm==4.64.1 \
    && conda clean -ya

RUN pip3 install -U --no-cache-dir \
    matplotlib==3.6.2

COPY postInstall /
RUN /postInstall
