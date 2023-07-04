FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime AS processor

# Install the additional Python dependencies
USER root
RUN python -m pip install --upgrade --no-cache-dir pip==23.1.2
RUN python -m pip install --upgrade --no-cache-dir scikit-image==0.21.0 scipy==1.11.1 pandas==2.0.3 SimpleITK==2.2.1

# Create user and project folder for user
RUN useradd -ms /bin/bash user
RUN mkdir /home/user/DSB2017
RUN chown user:user -R /home/user/DSB2017
WORKDIR /home/user

# Copy the model and code
USER user
COPY ["model/", "/home/user/DSB2017/model/"]
COPY ["preprocessing/", "/home/user/DSB2017/preprocessing/"]
COPY ["*.py", "/home/user/DSB2017/"]
COPY ["xmlreport.py", "/home/user/"]
COPY ["processor/apply.py", "/home/user/"]

# Set the ENV GIT_COMMIT_ID so that python can use this to insert in the xmlreport.py
ARG GIT_COMMIT_ID=unspecified
ENV GIT_COMMIT_ID=$GIT_COMMIT_ID

# first ensure that xmlreport.py matches the md5 hash version,
# subsequently set the GIT_COMMIT_ID in an artificial git folder for lookup
# expects line endings to be LF
RUN echo "473fe3ab2d87a6f06f569c04ecf5f0c2  /home/user/DSB2017/xmlreport.py" | md5sum -c -
RUN mkdir /home/user/DSB2017/.git && echo "$GIT_COMMIT_ID" > /home/user/DSB2017/.git/HEAD
ENTRYPOINT ["python", "apply.py", "--n-gpus=1", "--n-cpus=8"]

# Set some labels
LABEL nl.diagnijmegen.rse.algorithm.name=grt123
LABEL nl.diagnijmegen.rse.algorithm.git_commit_id=$GIT_COMMIT_ID


FROM processor AS test

USER root
RUN pip install pytest pytest-cov
RUN apt-get update && apt-get install -y liblzma-dev && rm -rf /var/lib/apt/lists/*
USER user
