#!/usr/bin/env bash

if [ $# -eq 0 ]; then
  ./build_processor_docker.sh
fi
TESTDATADIR=$(realpath ./tests/resources/inputs)
TMPDIR=$(mktemp -d -t test-processor-XXXXXX)
echo "$TESTDATADIR"
echo "$TMPDIR"
docker run -it --rm -v "$TESTDATADIR:/input" -v "$TMPDIR:/output" oni:11500/uoks/bodyct-kaggle-grt123:processor
rm -rf "$TMPDIR"
echo "Test successful!"
