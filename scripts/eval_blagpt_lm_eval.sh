#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HARNESS_ROOT="${REPO_ROOT}/lm-evaluation-harness"

MAIN_TASKS="${MAIN_TASKS:-mmlu,arc_challenge,hellaswag,truthfulqa_mc2,winogrande,gsm8k,piqa,arc_easy}"
LONG_CONTEXT_TASKS="${LONG_CONTEXT_TASKS:-longbench_triviaqa,longbench_qasper,infinitebench_passkey,longbench2_govt_single}"
TASKS="${TASKS:-${MAIN_TASKS},${LONG_CONTEXT_TASKS}}"
LENGTHS="${LENGTHS:-1024 2048 4096 8192}"
ATTENTIONS="${ATTENTIONS:-mamba2_GQA mamba2 regular flash_swa}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
DTYPE="${DTYPE:-bfloat16}"
MODEL_NAME="${MODEL_NAME:-blagpt}"
DEVICE="${DEVICE:-cuda}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
NUM_GPUS="${NUM_GPUS:-4}"
PARALLEL_JOBS="${PARALLEL_JOBS:-${NUM_GPUS}}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/logs/lm_eval}"
RULER_TOKENIZER="${RULER_TOKENIZER:-gpt2}"
HF_HOME="${HF_HOME:-/data/yinghaoliu/cache/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE:-${HF_HOME}}}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HUB_CACHE}}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
http_proxy="${http_proxy:-http://127.0.0.1:7897}"
https_proxy="${https_proxy:-http://127.0.0.1:7897}"
HTTP_PROXY="${HTTP_PROXY:-${http_proxy}}"
HTTPS_PROXY="${HTTPS_PROXY:-${https_proxy}}"

MAMBA2_GQA_CKPT="${MAMBA2_GQA_CKPT:-${REPO_ROOT}/logs/mamba2_new_v3_w512_2/best_model_78242.pt}"
MAMBA2_CKPT="${MAMBA2_CKPT:-${REPO_ROOT}/logs/mamba2_w512_0/best_model_78242.pt}"
REGULAR_CKPT="${REGULAR_CKPT:-${REPO_ROOT}/logs/regular_0/best_model_78242.pt}"
FLASH_SWA_CKPT="${FLASH_SWA_CKPT:-${REPO_ROOT}/logs/flash_swa512_5/best_model_78242.pt}"

if [[ -z "${FLASH_SWA_CKPT}" ]]; then
  FLASH_SWA_CKPT="$(find "${REPO_ROOT}/logs" -maxdepth 2 -type f -path '*flash*' -name 'best_model_*.pt' | sort | tail -n 1 || true)"
fi

cd "${HARNESS_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/bla_gpt:${HARNESS_ROOT}:${PYTHONPATH:-}"
export HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE HF_DATASETS_CACHE HF_ENDPOINT
export http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

metadata_for_length() {
  local max_length="$1"
  if [[ -n "${METADATA:-}" ]]; then
    printf '%s\n' "${METADATA}"
  else
    printf '{"tokenizer":"%s","max_seq_lengths":[%s]}\n' "${RULER_TOKENIZER}" "${max_length}"
  fi
}

ckpt_for_attention() {
  case "$1" in
    mamba2_GQA) printf '%s\n' "${MAMBA2_GQA_CKPT}" ;;
    mamba2) printf '%s\n' "${MAMBA2_CKPT}" ;;
    regular) printf '%s\n' "${REGULAR_CKPT}" ;;
    flash_swa) printf '%s\n' "${FLASH_SWA_CKPT}" ;;
    *) return 1 ;;
  esac
}

run_eval_job() {
  local attention="$1"
  local max_length="$2"
  local ckpt="$3"
  local gpu_id="$4"
  local out_dir="${OUT_ROOT}/${attention}/${max_length}"
  local log_file="${out_dir}/run.log"
  local metadata
  metadata="$(metadata_for_length "${max_length}")"

  mkdir -p "${out_dir}"
  echo "[eval] gpu=${gpu_id} attention=${attention} length=${max_length} ckpt=${ckpt}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    eval_args=(
      python -m lm_eval run
      --model blagpt \
      --model_args "checkpoint=${ckpt},model_name=${MODEL_NAME},max_length=${max_length},dtype=${DTYPE}" \
      --tasks "${TASKS}" \
      --num_fewshot "${NUM_FEWSHOT}" \
      --batch_size "${BATCH_SIZE}" \
      --device "${DEVICE}" \
      --output_path "${out_dir}"
    )
    if [[ -n "${metadata}" ]]; then
      eval_args+=(--metadata "${metadata}")
    fi
    "${eval_args[@]}"
  ) >"${log_file}" 2>&1
}

pids=()
job_index=0

wait_for_batch() {
  local pid
  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  pids=()
  if [[ "${failed}" -ne 0 ]]; then
    echo "[error] one or more eval jobs failed. Check run.log files under ${OUT_ROOT}."
    exit 1
  fi
}

for attention in ${ATTENTIONS}; do
  ckpt="$(ckpt_for_attention "${attention}")"
  if [[ -z "${ckpt}" || ! -f "${ckpt}" ]]; then
    echo "[skip] ${attention}: checkpoint not found. Set ${attention^^}_CKPT to include it."
    continue
  fi

  for max_length in ${LENGTHS}; do
    gpu_id=$((job_index % NUM_GPUS))
    run_eval_job "${attention}" "${max_length}" "${ckpt}" "${gpu_id}" &
    pids+=("$!")
    job_index=$((job_index + 1))

    if [[ "${#pids[@]}" -ge "${PARALLEL_JOBS}" ]]; then
      wait_for_batch
    fi
  done
done

if [[ "${#pids[@]}" -gt 0 ]]; then
  wait_for_batch
fi

echo "[done] all eval jobs finished. Results are under ${OUT_ROOT}."
