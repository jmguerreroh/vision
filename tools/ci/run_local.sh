#!/bin/bash
# Runs the CI on this machine, in the same images GitHub Actions uses.
#
#   tools/ci/run_local.sh                          # 24.04, 26.04, jazzy and lyrical
#   tools/ci/run_local.sh 26.04                    # only one target
#   tools/ci/run_local.sh 24.04 --steps build,run --only '^15_'
#
# Arguments that are not a target go to the script of each target
# (run_examples.py or ros2_check.py; see --help of each). The working tree is
# mounted read only and tested as it is, uncommitted changes included: the
# scripts copy it into the container before building anything.

set -e

ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
CI="$ROOT/tools/ci"

targets=()
extra=()
for arg in "$@"; do
  case "$arg" in
    24.04|26.04|jazzy|lyrical) targets+=("$arg") ;;
    *) extra+=("$arg") ;;
  esac
done
[ ${#targets[@]} -eq 0 ] && targets=(24.04 26.04 jazzy lyrical)

status=0
for target in "${targets[@]}"; do
  echo "##### $target"
  case "$target" in
    jazzy|lyrical)
      docker build -q -f "$CI/Dockerfile.ros2" --build-arg ROS_DISTRO="$target" \
        -t "vision-ci:$target" "$CI" > /dev/null
      script=ros2_check.py
      title="ROS 2 $target"
      ;;
    *)
      docker build -q -f "$CI/Dockerfile" --build-arg UBUNTU="$target" \
        -t "vision-ci:$target" "$CI" > /dev/null
      script=run_examples.py
      title="Ubuntu $target"
      ;;
  esac
  docker run --rm --init -v "$ROOT":/src:ro "vision-ci:$target" \
    xvfb-run -a -s "-screen 0 1920x1080x24" \
    python3 "/src/tools/ci/$script" --src /src --work /work --title "$title" "${extra[@]}" \
    || status=1
done
exit $status
