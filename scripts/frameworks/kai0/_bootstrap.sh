#!/usr/bin/env bash

resolve_project_python() {
  local project_root="$1"
  local backend_config="${2:-${project_root}/configs/frameworks/kai0/base/kai0_jax.yaml}"
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

export_robo_train_pythonpath() {
  local project_root="$1"
  export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
}

normalize_cuda_visible_devices() {
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" && -n "${CUDA_VISIBEL_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBEL_DEVICES}"
  fi
}

is_single_cuda_device() {
  normalize_cuda_visible_devices
  local devices="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
  [[ "${devices}" != *,* ]]
}

visible_cuda_device_count() {
  normalize_cuda_visible_devices
  local devices="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
  local count=0
  local device
  IFS=',' read -ra _kai0_visible_devices <<< "${devices}"
  for device in "${_kai0_visible_devices[@]}"; do
    device="${device//[[:space:]]/}"
    if [[ -n "${device}" ]]; then
      count=$((count + 1))
    fi
  done
  if [[ "${count}" -lt 1 ]]; then
    count=1
  fi
  echo "${count}"
}

has_cli_arg() {
  local name="$1"
  shift
  local arg
  for arg in "$@"; do
    if [[ "${arg}" == "${name}" || "${arg}" == "${name}="* ]]; then
      return 0
    fi
  done
  return 1
}

append_table30v2_card_defaults() {
  local single_run_name="$1"
  local multi_run_name="$2"
  shift 2

  normalize_cuda_visible_devices
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

  local device_count
  device_count="$(visible_cuda_device_count)"
  local default_run_name="${multi_run_name}"
  local default_batch_size="256"
  local default_num_workers="32"
  if is_single_cuda_device; then
    default_run_name="${single_run_name}"
    default_batch_size="8"
    default_num_workers="2"
  elif [[ "${device_count}" != "8" ]]; then
    default_batch_size="$((device_count * 32))"
    default_run_name="${multi_run_name/8card-bs256/${device_count}card-bs${default_batch_size}}"
    default_num_workers="8"
  fi

  local run_name="${RUN_NAME:-${default_run_name}}"
  local batch_size="${BATCH_SIZE:-${default_batch_size}}"
  local num_workers="${NUM_WORKERS:-${default_num_workers}}"

  TABLE30V2_DEFAULT_ARGS=()
  if ! has_cli_arg "--exp_name" "$@" && ! has_cli_arg "--exp-name" "$@"; then
    TABLE30V2_DEFAULT_ARGS+=(--exp_name "${run_name}")
  fi
  if ! has_cli_arg "--batch_size" "$@" && ! has_cli_arg "--batch-size" "$@"; then
    TABLE30V2_DEFAULT_ARGS+=(--batch_size "${batch_size}")
  fi
  if ! has_cli_arg "--num_workers" "$@" && ! has_cli_arg "--num-workers" "$@"; then
    TABLE30V2_DEFAULT_ARGS+=(--num_workers "${num_workers}")
  fi
}
