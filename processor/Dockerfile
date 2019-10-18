FROM oni:11500/uokbaseimage/tensorflow_pytorch_diag_dicomloader_python2:2 AS processor

USER user
COPY ["processor/Pipfile", "processor/Pipfile.lock", "processor/apply.py", "/home/user/"]
COPY [".", "/home/user/DSB2017/"]

USER root
RUN chown user:user -R /home/user/DSB2017

WORKDIR /home/user
RUN LC_ALL=C.UTF-8 pip install pipenv
RUN LC_ALL=C.UTF-8 pipenv --python 2.7 install --system && rm -rf /root/.cache

USER user
ENTRYPOINT ["/usr/bin/python2", "apply.py", "--n-gpus=1", "--n-cpus=8"]

LABEL com.radboudumc.diag.processor.name=grt123
LABEL com.radboudumc.diag.processor.ticket=8765

LABEL com.radboudumc.diag.processor.hardware.cpu.count=8
LABEL com.radboudumc.diag.processor.hardware.cpu.capabilities=
LABEL com.radboudumc.diag.processor.hardware.memory=30G
LABEL com.radboudumc.diag.processor.hardware.gpu.count=1
LABEL com.radboudumc.diag.processor.hardware.gpu.cuda_compute_capability=
LABEL com.radboudumc.diag.processor.hardware.gpu.memory=4G

ARG GIT_COMMIT_ID=unspecified
LABEL com.radboudumc.diag.processor.git_commit_id=$GIT_COMMIT_ID


FROM processor AS test

USER root

RUN pip install pytest pytest-cov tox twine

RUN apt-get install liblzma-dev -y
RUN pip install backports.lzma

USER user
