# DSB2017 challenge: grt123 processor

This repository uses a modifed grt123 solution to implement a DIAG-processor docker image. The processor can be applied to a list of 3D lung images and the algorithm will:

* Compute cancer scores

    Cancer scores indicate the likelyhood of the image containing a cancer.

* Designate regions of interest

    Regions of interest are locations in the input image that the algorithm analyzed to compute its "cancer score".

## Building

The processor uses docker to build a containerized runtime environment for the grt123 algorithm to run in. To build the runtime environment, DIAG's `oni:11500` docker registry must be avaialbe, since it uses the uok-base images stored there.

If you are in the DIAG-network, issue the command

```
docker build -t grt123_processor .
```

to build the docker image.


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

The input directory is a directory lung image volumes. Images are either:

- subdirectories containing slice-based DICOM images
- MHA or MHD+ZRAW MetaIO files containing image volumes

### Output

(Forgot the details...)

- A CSV file with data
- A JSON file with more data
