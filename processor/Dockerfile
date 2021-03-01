FROM doduo1.umcn.nl/uokbaseimage/tensorflow_pytorch_diag_dicomloader_python2:2 AS processor

# Install the Python environment
USER root
COPY ["processor/Pipfile", "processor/Pipfile.lock", "/home/user/"]
WORKDIR /home/user
RUN LC_ALL=C.UTF-8 pip install pipenv
RUN LC_ALL=C.UTF-8 pipenv --python 2.7 install --system && rm -rf /root/.cache

# Copy the model and code
USER user
COPY ["model/", "/home/user/DSB2017/model/"]
COPY ["preprocessing/", "/home/user/DSB2017/preprocessing/"]
COPY ["*.py", "/home/user/DSB2017/"]
COPY ["xmlreport.py", "/home/user/"]
COPY ["processor/apply.py", "/home/user/"]

ARG GIT_COMMIT_ID=unspecified
# Set the ENV GIT_COMMIT_ID so that python can use this to insert in the xmlreport.py
ENV GIT_COMMIT_ID=$GIT_COMMIT_ID

# first ensure that xmlreport.py matches the md5 hash version,
# subsequently replace the GIT_COMMIT_ID lookup function with the embeded GIT_COMMIT_ID string
# expects line endings to be LF
RUN echo "419020b0fad961904cd6f20fd4a073c9  /home/user/DSB2017/xmlreport.py" | md5sum -c -
RUN python -c "fn='/home/user/DSB2017/xmlreport.py'; f=open(fn, 'r'); data=f.readlines(); f.close(); data=data[:10] + ['    return \"$GIT_COMMIT_ID\"\n'] + data[15:]; f=open(fn, 'w'); f.writelines(data); f.close();"

USER root
RUN chown user:user -R /home/user/DSB2017

USER user
ENTRYPOINT ["/usr/bin/python2", "apply.py", "--n-gpus=1", "--n-cpus=8"]

LABEL nl.diagnijmegen.rse.algorithm.name=grt123
LABEL nl.diagnijmegen.rse.algorithm.ticket=8765

LABEL nl.diagnijmegen.rse.algorithm.hardware.cpu.count=8
LABEL nl.diagnijmegen.rse.algorithm.hardware.cpu.capabilities=""
LABEL nl.diagnijmegen.rse.algorithm.hardware.memory=30G
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.count=1
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.cuda_compute_capability=""
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.memory=4G

LABEL nl.diagnijmegen.rse.algorithm.git_commit_id=$GIT_COMMIT_ID



FROM processor AS test

USER root
RUN pip install pytest pytest-cov tox twine
RUN apt-get install liblzma-dev -y
RUN pip install backports.lzma
USER user
