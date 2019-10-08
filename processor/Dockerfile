FROM oni:11500/uokbaseimage/tensorflow_pytorch_diag_dicomloader_python2:2

USER user
COPY ["processor/Pipfile", "processor/Pipfile.lock", "processor/apply.py", "/home/user/"]
COPY [".", "/home/user/DSB2017"]

USER root
RUN chown user:user -R /home/user/DSB2017

WORKDIR /home/user
RUN LC_ALL=C.UTF-8 pip install pipenv
RUN LC_ALL=C.UTF-8 pipenv --python 2.7 install --system && rm -rf /root/.cache

USER user
ENTRYPOINT ["/usr/bin/python2", "apply.py", "--n-gpus=1", "--n-cpus=8"]

LABEL processor.hardware.cpus=8
LABEL processor.hardware.memory=30G
LABEL processor.hardware.gpus=1
LABEL processor.hardware.gpu.type=
LABEL processor.hardware.gpu.memory=4G

# Add Git commit ID of current codebase
ARG GIT_COMMIT_ID=unspecified
LABEL processor.git_commit_id=$GIT_COMMIT_ID
