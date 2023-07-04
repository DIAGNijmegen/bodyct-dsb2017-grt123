#!/usr/bin/env bash

display_help () {
  echo "usage: $0 [1 or more IMAGES] [--version-tag VERSION_TAG] [--git-commit GIT_COMMIT_ID] [--push] [-h|--help]"
  echo "IMAGES: processor processor-test"
  echo "VERSION_TAG: specify the full version tag (i.e. prepend a \"v\"),"
  echo "for example: \"v2.2\"."
  exit 1
}

PUSH_IMAGES=False
VERSION_TAG=""
GIT_COMMIT_ID=$(cat .git/HEAD)
if [[ $GIT_COMMIT_ID == "ref:"* ]] ; then
  GIT_COMMIT_ID=$(cat .git/$(cat .git/HEAD | awk '{ print $2 }'))
fi

declare -a BUILD_LIST=()

while [ $# -ne 0 ]
do
  case "$1" in
    processor)
      BUILD_LIST+=("$1")
      ;;
    processor-test)
      BUILD_LIST+=("$1")
      ;;
    --version-tag)
      shift
      if [ $# -eq 0 ] ; then
        echo "--version-tag must be followed by a string"
        exit 1
      else
        VERSION_TAG="$1"
      fi
      ;;
    --push)
      PUSH_IMAGES=True
      ;;
    --git-commit)
      shift
      if [ $# -eq 0 ] ; then
        echo "--git-commit must be followed by a string"
        exit 1
      else
        GIT_COMMIT_ID="$1"
      fi
      ;;
    --help)
      display_help
      ;;
    -h)
      display_help
      ;;
    *)
      echo "invalid option: $1"
      exit 0
      ;;
  esac
  shift
done

if [ -n "$VERSION_TAG" ] ; then
  echo "Version tag was set to: $VERSION_TAG"
else
  echo "No version tag specified."
fi

echo "GIT COMMIT ID: $GIT_COMMIT_ID"

for build_item in "${BUILD_LIST[@]}";
do
  echo "ATTEMPT BUILDING: $build_item"
  case "$build_item" in
    processor)
      if [ -n "$VERSION_TAG" ] ; then
        docker build --tag=doduo1.umcn.nl/bodyct/releases/bodyct-kaggle-grt123:$VERSION_TAG -f ./processor/Dockerfile --target=processor --build-arg GIT_COMMIT_ID=$GIT_COMMIT_ID .
        if [ $PUSH_IMAGES = True ] ; then
          docker push doduo1.umcn.nl/bodyct/releases/bodyct-kaggle-grt123:$VERSION_TAG
          docker tag doduo1.umcn.nl/bodyct/releases/bodyct-kaggle-grt123:$VERSION_TAG doduo1.umcn.nl/bodyct/releases/bodyct-multiview-nodule-detection:latest
          docker push doduo1.umcn.nl/uoks/bodyct-multiview-nodule-detection:latest
        fi
      else
        docker build --tag=doduo1.umcn.nl/bodyct/releases/bodyct-kaggle-grt123:latest -f ./processor/Dockerfile --target=processor --build-arg GIT_COMMIT_ID=$GIT_COMMIT_ID .
        if [ $PUSH_IMAGES = True ] ; then
          docker push doduo1.umcn.nl/uoks/bodyct-multiview-nodule-detection:latest
        fi
      fi
      ;;
    processor-test)
      docker build --tag=doduo1.umcn.nl/bodyct/releases/bodyct-kaggle-grt123:test$VERSION_TAG -f ./processor/Dockerfile --target=test --build-arg GIT_COMMIT_ID=$GIT_COMMIT_ID .
      ;;
    *)
      echo "invalid build-item: $build_item"
      exit 1
  esac
done
exit 0
