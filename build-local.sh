#!/usr/bin/env bash
#
# Build the prod images the same way .github/workflows/build-push-prod.yml does,
# but push them to the local insecure registry instead of GHCR.
#
# Usage:
#   ./build-local.sh                  # build + push all services
#   ./build-local.sh api frontend     # build + push a subset
#
# Env overrides:
#   REGISTRY       registry host:port            (default datavaultb360-hades:5000)
#   IMAGE_PREFIX   image name prefix             (default cv5)
#   VITE_API_URL   baked into the frontend build
#   GH_TOKEN       passed as BuildKit secret `gh_token` to the api build (model download)
#   NO_PUSH=1      build only, skip the push

set -euo pipefail

REGISTRY="${REGISTRY:-datavaultb360-hades:5000}"
IMAGE_PREFIX="${IMAGE_PREFIX:-cv5}"
VITE_API_URL="${VITE_API_URL:-http://datavaultb360-hades:2024/api/v1}"

# name|dockerfile|context
SERVICES=(
  "migrations|db/Dockerfile|."
  "api|api/Dockerfile.local|."
  "frontend|frontend/Dockerfile|."
)

export DOCKER_BUILDKIT=1

cd "$(git rev-parse --show-toplevel)"

SHA="$(git rev-parse --short HEAD)"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "WARNING: working tree is dirty; sha-${SHA} will not match the committed source" >&2
fi

echo "==> Checking registry ${REGISTRY}"
curl -fsS "http://${REGISTRY}/v2/" >/dev/null || {
  echo "ERROR: cannot reach http://${REGISTRY}/v2/" >&2
  exit 1
}

build_one() {
  local name="$1" file="$2" context="$3"
  local base="${REGISTRY}/${IMAGE_PREFIX}-${name}"
  local args=(--file "$file" --tag "${base}:latest" --tag "${base}:sha-${SHA}")

  case "$name" in
    frontend)
      args+=(--build-arg "VITE_API_URL=${VITE_API_URL}")
      ;;
    api)
      if [[ -n "${GH_TOKEN:-}" ]]; then
        args+=(--secret "id=gh_token,env=GH_TOKEN")
      fi
      ;;
  esac

  echo "==> Building ${base}:sha-${SHA}"
  docker build "${args[@]}" "$context"

  if [[ "${NO_PUSH:-0}" == "1" ]]; then
    echo "==> NO_PUSH set, skipping push for ${name}"
    return
  fi

  echo "==> Pushing ${base}"
  docker push "${base}:latest"
  docker push "${base}:sha-${SHA}"
}

# Resolve which services to build
targets=("$@")
if [[ ${#targets[@]} -eq 0 ]]; then
  for entry in "${SERVICES[@]}"; do
    targets+=("${entry%%|*}")
  done
fi

built=()
for target in "${targets[@]}"; do
  found=0
  for entry in "${SERVICES[@]}"; do
    IFS='|' read -r name file context <<<"$entry"
    if [[ "$name" == "$target" ]]; then
      build_one "$name" "$file" "$context"
      built+=("${REGISTRY}/${IMAGE_PREFIX}-${name}:sha-${SHA}")
      found=1
      break
    fi
  done
  if [[ $found -eq 0 ]]; then
    echo "ERROR: unknown service '${target}'" >&2
    exit 1
  fi
done

echo
echo "Done. Images:"
printf '  %s\n' "${built[@]}"