#!/usr/bin/env bash

if [ $# -eq 0 ] ; then
  docker build -f ./processor/Dockerfile --target=processor --tag=doduo1.umcn.nl/uoks/bodyct-kaggle-grt123:processor --build-arg GIT_COMMIT_ID=$(cat .git/$(cat .git/HEAD | awk '{ print $2 }')) .
else
  if [ "$1" == "--test" ] ; then
      docker build -f ./processor/Dockerfile --target=test --tag=doduo1.umcn.nl/uoks/bodyct-kaggle-grt123:test --build-arg GIT_COMMIT_ID=$(cat .git/$(cat .git/HEAD | awk '{ print $2 }')) .
  else
      echo "usage: $0 [--test]"
  fi
fi
