#!/usr/bin/env bash
docker build -f ./processor/Dockerfile --tag=oni:11500/uoks/bodyct-kaggle-grt123:processor --build-arg GIT_COMMIT_ID=$(cat .git/$(cat .git/HEAD | awk '{ print $2 }')) .
