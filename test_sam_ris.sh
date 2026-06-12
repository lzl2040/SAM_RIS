#!/usr/bin/env bash
set -euo pipefail

check_required_values() {
  local previous=""
  for arg in "$@"; do
    if [[ -n "${previous}" ]]; then
      if [[ "${arg}" == --* ]]; then
        echo "Missing value for ${previous}" >&2
        exit 2
      fi
      previous=""
    fi

    case "${arg}" in
      --sam_model_type|--sam_pretrain_path|--dataset|--split|--splitBy|--resume)
        previous="${arg}"
        ;;
    esac
  done

  if [[ -n "${previous}" ]]; then
    echo "Missing value for ${previous}" >&2
    exit 2
  fi
}

check_required_values "$@"

python test_sam_ris.py "$@"
