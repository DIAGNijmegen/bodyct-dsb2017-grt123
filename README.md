# DSB2017 challenge: grt123 processor

This repository uses a modified grt123 solution to implement a DIAG-processor docker image. The processor can be applied to a list of 3D lung images and the algorithm will:

* Compute cancer scores

    Cancer scores indicate the likelihood of the image containing a cancer.

* Designate regions of interest

    Regions of interest are locations in the input image that the algorithm analyzed to compute its "cancer score".

Original grt123 solution can be found [here](https://github.com/lfz/DSB2017).

## Building

The processor uses docker to build a containerized runtime environment for the grt123 algorithm to run in. 

If you are in a linux environment with docker installed, issue the command:

```
./build_processor_docker.sh processor [--version-tag VERSION_TAG] [--git-commit GIT_COMMIT_ID] [--push] [-h|--help]
```

to build the DIAG docker image.

* `--version-tag VERSION_TAG` labels the docker image with a version tag, if not specified the tag is left empty creating `latest`.
* `--git-commit GIT_COMMIT_ID` overrides the internally baked gid commit id used by the processor, if not specified will attempt to look for it in the `.git` folder.
* `--push` will attempt to push the images to the private DIAG-docker-registry after building. 
* `-h, --help` will display some a short help message.


## Running

The algorithm requires the following hardware configuration to run:

- 8 CPUs
- 30GB of memory
- 1 CUDA capable GPU with at least 4 GB of memory

The measured runtime for the algorithm on a system matching the specifications is:

- 40 seconds startup time plus ...
    - ... 40 seconds per scan when processing many scans at once
    - ... 60 seconds when processing single scans

### Input

The input directory is a directory lung image volumes. Images are:

- MHA or MHD+ZRAW MetaIO files containing image volumes

### Output

- Per processed image an XML file with candidates and cancer scores
- A JSON file summarizing all findings
