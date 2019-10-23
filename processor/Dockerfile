FROM oni:11500/uokbaseimage/tensorflow_pytorch_diag_dicomloader_python2:2 AS processor

ARG GIT_COMMIT_ID=unspecified

USER user
COPY ["processor/Pipfile", "processor/Pipfile.lock", "processor/apply.py", "/home/user/"]
COPY ["*.py", "/home/user/DSB2017/"]
COPY ["model/", "/home/user/DSB2017/model/"]
COPY ["preprocessing/", "/home/user/DSB2017/preprocessing/"]

# first ensure that xmlreport.py matches the md5 hash version,
# subsequently replace the GIT_COMMIT_ID lookup function with the embeded GIT_COMMIT_ID string
# expects line endings to be LF
RUN echo "ecaf1db67f153ec12f7068cc84eb1da6  /home/user/DSB2017/xmlreport.py" | md5sum -c -
RUN python -c "fn='/home/user/DSB2017/xmlreport.py'; f=open(fn, 'r'); data=f.readlines(); f.close(); data=data[:9] + ['    return \"$GIT_COMMIT_ID\"\n'] + data[14:]; f=open(fn, 'w'); f.writelines(data); f.close();"

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

LABEL com.radboudumc.diag.processor.git_commit_id=$GIT_COMMIT_ID


FROM processor AS test

USER root

RUN pip install pytest pytest-cov tox twine

RUN apt-get install liblzma-dev -y
RUN pip install backports.lzma

USER user
