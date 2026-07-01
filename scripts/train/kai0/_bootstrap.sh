#!/usr/bin/env bash

resolve_project_python() {
  local project_root="$1"
  local backend_config="${2:-${project_root}/configs/base/train/kai0_jax.yaml}"
  local python_bin="${PYTHON_BIN:-${project_root}/.venv/bin/python}"

  if [[ -x "${python_bin}" ]]; then
    echo "${python_bin}"
    return 0
  fi

  if [[ -n "${KAI0_PYTHON:-}" && -x "${KAI0_PYTHON}" ]]; then
    echo "${KAI0_PYTHON}"
    return 0
  fi

  if [[ -f "${backend_config}" ]]; then
    local source_root
    local rel_python
    source_root="$(awk '/^[[:space:]]*source_root:/ {print $2; exit}' "${backend_config}")"
    rel_python="$(awk '/^[[:space:]]*python_bin:/ {print $2; exit}' "${backend_config}")"
    if [[ -n "${source_root}" && -n "${rel_python}" && -x "${source_root}/${rel_python}" ]]; then
      echo "${source_root}/${rel_python}"
      return 0
    fi
  fi

  if [[ -n "${CONDA_PYTHON:-}" && -x "${CONDA_PYTHON}" ]]; then
    echo "${CONDA_PYTHON}"
    return 0
  fi

  if [[ -x /opt/conda/bin/python ]]; then
    echo /opt/conda/bin/python
    return 0
  fi

  echo "${PYTHON_BIN_FALLBACK:-python}"
}

kai0_table30v2_args() {
  local default_run_name="$1"
  shift
  local -n out_args="$1"
  shift

  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" && -n "${CUDA_VISIBEL_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBEL_DEVICES}"
  fi

  local has_exp_name=0
  local has_batch_size=0
  local arg
  for arg in "$@"; do
    case "${arg}" in
      --exp_name|--exp-name|--exp_name=*|--exp-name=*)
        has_exp_name=1
        ;;
      --batch_size|--batch-size|--batch_size=*|--batch-size=*)
        has_batch_size=1
        ;;
    esac
  done

  local cuda_devices="${CUDA_VISIBLE_DEVICES:-}"
  local run_name="${default_run_name}"
  local single_card=0
  if [[ -n "${cuda_devices}" && "${cuda_devices}" != *,* ]]; then
    single_card=1
    run_name="${run_name/8card-bs256/1card-bs8}"
  fi

  out_args=()
  if [[ "${has_exp_name}" == "0" ]]; then
    out_args+=(--exp_name "${run_name}")
  fi
  if [[ "${single_card}" == "1" && "${has_batch_size}" == "0" ]]; then
    out_args+=(--batch_size 8)
  fi
}
