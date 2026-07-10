#!/usr/bin/env bash
#
# ADFNet Ablation Study - Batch Parallel Launcher
# =================================================
#
# 将消融实验以 nohup 方式并行提交到多张 GPU 上。
# 每个消融配置 × 每种任务难度 = 一个独立后台进程（内含 kfold + loso）。
#
# 用法:
#   bash scripts/run_all_ablations.sh [模式] [选项]
#
# 模式:
#   all              全部 64 种组合 + LSTM/Transformer 替换 (默认)
#   single <preset>  单个预设 (full/no_gamma/no_grl/...)
#   combinations     仅 64 种组合
#   lstm             LSTM 替换 × 5 个其他开关的 32 种组合
#   transformer      Transformer 替换 × 5 个其他开关的 32 种组合
#   status           查看运行状态
#
# 选项:
#   --gpus "0 1 2 3"       GPU 编号 (空格分隔, 默认自动检测)
#   --cv kfold|loso|both   交叉验证 (默认 both)
#   --task easy|hard|both  任务难度 (默认 both)
#   --max-parallel N       每张 GPU 最大并行数 (默认 2)
#   --max-folds N          调试用: 每种 CV 只跑 N 个 fold
#   --config path          配置文件 (默认 configs/default.yaml)
#   --output-dir path      输出根目录 (默认 ./outputs/ablation)
#
# 示例:
#   # 全部实验, 4 卡并行
#   bash scripts/run_all_ablations.sh all --gpus "0 1 2 3"
#
#   # 只跑 no_grl, 仅 kfold + easy (快速验证)
#   bash scripts/run_all_ablations.sh single no_grl --cv kfold --task easy
#
#   # 全部 64 组合, 每卡只跑 1 个 fold (冒烟测试)
#   bash scripts/run_all_ablations.sh combinations --max-folds 1
#
#   # 查看进度
#   bash scripts/run_all_ablations.sh status
#
# 手动 nohup 单个实验:
#   CUDA_VISIBLE_DEVICES=0 nohup python scripts/run_loso.py \
#     --config configs/default.yaml \
#     --task-mode easy \
#     --ablation enable_grl=false \
#     --exp-name ablation_no_grl \
#     --output-dir ./outputs/ablation/no_grl/loso_easy \
#     > logs/no_grl_loso_easy.log 2>&1 &
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ===================== 默认配置 (可由命令行覆盖) =====================
GPUS=""                     # 留空则自动检测
CV="both"                   # kfold | loso | both
TASK_MODES="both"           # easy | hard | both
MAX_PARALLEL_PER_GPU=2
CONFIG="configs/default.yaml"
OUTPUT_DIR="./outputs/ablation"
MAX_FOLDS=""

# ===================== 解析命令行参数 =====================
MODE="all"
SINGLE_PRESET=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        all|combinations|lstm|transformer|gaussian|kde|status)
            MODE="$1"; shift ;;
        single)
            MODE="single"
            shift
            SINGLE_PRESET="${1:?single 模式需要指定预设名}"; shift ;;
        --gpus)       GPUS="$2"; shift 2 ;;
        --cv)         CV="$2"; shift 2 ;;
        --task)       TASK_MODES="$2"; shift 2 ;;
        --max-parallel) MAX_PARALLEL_PER_GPU="$2"; shift 2 ;;
        --config)     CONFIG="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --max-folds)  MAX_FOLDS="$2"; shift 2 ;;
        -h|--help)
            head -50 "$0" | tail -45
            exit 0 ;;
        *)
            echo "未知参数: $1"; exit 1 ;;
    esac
done

# ===================== GPU 检测 =====================
if [[ -z "$GPUS" ]]; then
    if command -v nvidia-smi &>/dev/null; then
        GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
        if [[ $GPU_COUNT -gt 0 ]]; then
            GPUS=$(seq 0 $((GPU_COUNT - 1)) | tr '\n' ' ')
        fi
    fi
fi
# 如果仍然为空 (无 GPU), 使用空字符串 (CPU 模式)
GPU_ARRAY=($GPUS)
NUM_GPUS=${#GPU_ARRAY[@]}
if [[ $NUM_GPUS -eq 0 ]]; then
    NUM_GPUS=1
    GPU_ARRAY=("")
    echo "[WARN] 未检测到 GPU, 将以 CPU 模式运行"
fi

# ===================== 任务难度展开 =====================
case "$TASK_MODES" in
    both) TASK_LIST=(easy hard) ;;
    easy) TASK_LIST=(easy) ;;
    hard) TASK_LIST=(hard) ;;
    *)    TASK_LIST=($TASK_MODES) ;;
esac

# ===================== CV 展开 =====================
case "$CV" in
    both) CV_LIST=(kfold loso) ;;
    *)    CV_LIST=($CV) ;;
esac

MAX_JOBS=$((NUM_GPUS * MAX_PARALLEL_PER_GPU))

# ===================== 预设定义 =====================
# 名称 -> ablation 参数
declare -A PRESET_ARGS=(
    [full]=""
    [no_gamma]="enable_gamma=false"
    [no_grl]="enable_grl=false"
    [no_diff]="enable_diff=false"
    [no_sliding_mean]="enable_sliding_mean=false"
    [no_soft_dtw]="enable_soft_dtw=false"
    [no_mamba]="enable_mamba=false"
    [gaussian]="reference_distribution=gaussian"
    [kde]="reference_distribution=kde"
)

# ===================== 日志目录 =====================
LOG_DIR="$OUTPUT_DIR/logs"

# ===================== 辅助函数 =====================
ts() { date '+%Y-%m-%d %H:%M:%S'; }

log() { echo "[$(ts)] $*"; }

# 生成 64 种组合: 输出 "ablation_args|label" 行
generate_combinations() {
    for gamma in true false; do
    for grl in true false; do
    for diff in true false; do
    for smean in true false; do
    for sdtw in true false; do
    for mamba in true false; do
        local args=""
        local label=""
        [[ "$gamma" == "false" ]] && args+="enable_gamma=false " && label+="_noGamma"
        [[ "$grl" == "false" ]]   && args+="enable_grl=false "   && label+="_noGRL"
        [[ "$diff" == "false" ]]  && args+="enable_diff=false "  && label+="_noDiff"
        [[ "$smean" == "false" ]] && args+="enable_sliding_mean=false " && label+="_noSMean"
        [[ "$sdtw" == "false" ]]  && args+="enable_soft_dtw=false "     && label+="_noSDTW"
        [[ "$mamba" == "false" ]] && args+="enable_mamba=false " && label+="_noMamba"
        label="${label#_}"
        [[ -z "$label" ]] && label="full"
        echo "${args}|${label}"
    done; done; done; done; done; done
}

# 生成替换实验: encoder_type × 5 个其他开关的 32 种组合
generate_replacements() {
    local encoder="$1"
    for gamma in true false; do
    for grl in true false; do
    for diff in true false; do
    for smean in true false; do
    for sdtw in true false; do
        local args="temporal_encoder=$encoder "
        local label="${encoder}"
        [[ "$gamma" == "false" ]] && args+="enable_gamma=false " && label+="_noGamma"
        [[ "$grl" == "false" ]]   && args+="enable_grl=false "   && label+="_noGRL"
        [[ "$diff" == "false" ]]  && args+="enable_diff=false "  && label+="_noDiff"
        [[ "$smean" == "false" ]] && args+="enable_sliding_mean=false " && label+="_noSMean"
        [[ "$sdtw" == "false" ]]  && args+="enable_soft_dtw=false "     && label+="_noSDTW"
        echo "${args}|${label}"
    done; done; done; done; done
}

# 生成分布替换实验: dist_type × 5 个其他开关的 32 种组合
generate_dist_replacements() {
    local dist_type="$1"
    for gamma in true false; do
    for grl in true false; do
    for diff in true false; do
    for smean in true false; do
    for sdtw in true false; do
        local args="reference_distribution=$dist_type "
        local label="${dist_type}"
        [[ "$gamma" == "false" ]] && args+="enable_gamma=false " && label+="_noGamma"
        [[ "$grl" == "false" ]]   && args+="enable_grl=false "   && label+="_noGRL"
        [[ "$diff" == "false" ]]  && args+="enable_diff=false "  && label+="_noDiff"
        [[ "$smean" == "false" ]] && args+="enable_sliding_mean=false " && label+="_noSMean"
        [[ "$sdtw" == "false" ]]  && args+="enable_soft_dtw=false "     && label+="_noSDTW"
        echo "${args}|${label}"
    done; done; done; done; done
}

# 全局作业索引 (用于 GPU round-robin)
JOB_INDEX=0
# 当前运行中的 PID 列表
declare -a RUNNING_PIDS=()
declare -a RUNNING_LABELS=()

# 提交一个 job, 如果超过并行限制则等待
launch_job() {
    local label="$1"
    local abl_args="$2"   # 可能为空

    for task in "${TASK_LIST[@]}"; do
        for cv in "${CV_LIST[@]}"; do
            # GPU 分配 (round-robin)
            local gpu_idx=$((JOB_INDEX % NUM_GPUS))
            local gpu_id="${GPU_ARRAY[$gpu_idx]}"
            JOB_INDEX=$((JOB_INDEX + 1))

            local exp_name="ablation_${label}_${cv}_${task}"
            local out_dir="${OUTPUT_DIR}/${label}/${cv}_${task}"
            local log_file="${LOG_DIR}/${label}_${cv}_${task}.log"

            # 构建命令
            local cmd="python scripts/run_${cv/run_}.py"
            # cv: kfold -> run_group_kfold.py, loso -> run_loso.py
            if [[ "$cv" == "kfold" ]]; then
                cmd="python scripts/run_group_kfold.py"
            else
                cmd="python scripts/run_loso.py"
            fi
            cmd+=" --config $CONFIG"
            cmd+=" --task-mode $task"
            cmd+=" --exp-name $exp_name"
            cmd+=" --output-dir $out_dir"
            if [[ -n "$abl_args" ]]; then
                cmd+=" --ablation $abl_args"
            fi
            if [[ -n "$MAX_FOLDS" ]]; then
                cmd+=" --max-folds $MAX_FOLDS"
            fi

            # 限流: 等待直到有空位
            while [[ ${#RUNNING_PIDS[@]} -ge $MAX_JOBS ]]; do
                _wait_for_slot
            done

            # 启动
            mkdir -p "$out_dir" "$(dirname "$log_file")"
            log "[$((JOB_INDEX))/${TOTAL_JOBS}] GPU=${gpu_id:-CPU} ${label} ${cv}/${task} -> $(basename "$log_file")"

            if [[ -n "$gpu_id" ]]; then
                CUDA_VISIBLE_DEVICES="$gpu_id" nohup $cmd > "$log_file" 2>&1 &
            else
                nohup $cmd > "$log_file" 2>&1 &
            fi

            local pid=$!
            RUNNING_PIDS+=($pid)
            RUNNING_LABELS+=("${label}_${cv}_${task}(pid=$pid)")
        done
    done
}

# 等待任一后台进程结束, 腾出空位
_wait_for_slot() {
    local new_pids=()
    local new_labels=()
    for i in "${!RUNNING_PIDS[@]}"; do
        if kill -0 "${RUNNING_PIDS[$i]}" 2>/dev/null; then
            new_pids+=("${RUNNING_PIDS[$i]}")
            new_labels+=("${RUNNING_LABELS[$i]}")
        fi
    done
    RUNNING_PIDS=("${new_pids[@]}")
    RUNNING_LABELS=("${new_labels[@]}")
    if [[ ${#RUNNING_PIDS[@]} -ge $MAX_JOBS ]]; then
        sleep 5
        _wait_for_slot  # 递归等待
    fi
}

# 等待所有后台进程完成
wait_all() {
    log "等待 ${#RUNNING_PIDS[@]} 个剩余任务完成..."
    for pid in "${RUNNING_PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    log "所有任务已完成"
}

# 显示运行状态
show_status() {
    echo "==================== ADFNet Ablation Status ===================="
    echo ""
    echo "日志目录: $LOG_DIR"
    echo ""
    if [[ ! -d "$LOG_DIR" ]]; then
        echo "  (无日志目录, 尚未运行过实验)"
        return
    fi

    local total=0
    local done_count=0
    local running_count=0
    local failed_count=0

    for log_file in "$LOG_DIR"/*.log; do
        [[ -f "$log_file" ]] || continue
        total=$((total + 1))
        local name=$(basename "$log_file" .log)
        # 检查是否有对应进程在跑
        local pid=$(pgrep -f "run_.*--exp-name.*${name%%_*}" 2>/dev/null | head -1 || true)
        if [[ -n "$pid" ]]; then
            running_count=$((running_count + 1))
            echo "  [RUNNING]  $name  (PID: $pid)"
        else
            # 检查日志末尾是否有 "Total training time"
            if tail -5 "$log_file" 2>/dev/null | grep -q "Total .* time"; then
                done_count=$((done_count + 1))
                local time_info=$(tail -5 "$log_file" | grep "Total" | head -1 | sed 's/.*Total/Total/')
                echo "  [DONE]     $name  $time_info"
            else
                # 检查是否有错误
                if grep -qi "error\|exception\|traceback" "$log_file" 2>/dev/null; then
                    failed_count=$((failed_count + 1))
                    echo "  [FAILED]   $name  (see $log_file)"
                else
                    done_count=$((done_count + 1))
                    echo "  [DONE]     $name"
                fi
            fi
        fi
    done
    echo ""
    echo "总计: $total  完成: $done_count  运行中: $running_count  失败: $failed_count"
    echo ""
    echo "快速检查:"
    echo "  查看某个日志:  tail -f $LOG_DIR/<name>.log"
    echo "  查看 GPU 占用:  nvidia-smi"
    echo "  查看 Python 进程: ps aux | grep python"
}

# ===================== 统计总 job 数 =====================
count_total_jobs() {
    local count=0
    for entry in "${JOB_ENTRIES[@]}"; do
        local n_cv=${#CV_LIST[@]}
        local n_task=${#TASK_LIST[@]}
        count=$((count + n_cv * n_task))
    done
    echo "$count"
}

# ===================== 模式: status =====================
if [[ "$MODE" == "status" ]]; then
    show_status
    exit 0
fi

# ===================== 收集所有实验条目 =====================
declare -a JOB_ENTRIES=()  # "ablation_args|label"

case "$MODE" in
    all)
        while IFS= read -r line; do
            JOB_ENTRIES+=("$line")
        done < <(generate_combinations)
        while IFS= read -r line; do
            JOB_ENTRIES+=("$line")
        done < <(generate_replacements "lstm")
        while IFS= read -r line; do
            JOB_ENTRIES+=("$line")
        done < <(generate_replacements "transformer")
        while IFS= read -r line; do
            JOB_ENTRIES+=("$line")
        done < <(generate_dist_replacements "gaussian")
        while IFS= read -r line; do
            JOB_ENTRIES+=("$line")
        done < <(generate_dist_replacements "kde")
        ;;
    combinations)
        while IFS= read -r line; do
            JOB_ENTRIES+=("$line")
        done < <(generate_combinations)
        ;;
    single)
        if [[ -z "$SINGLE_PRESET" ]]; then
            echo "错误: single 模式需要指定预设名"; exit 1
        fi
        local_args="${PRESET_ARGS[$SINGLE_PRESET]:-}"
        JOB_ENTRIES+=("${local_args}|${SINGLE_PRESET}")
        ;;
    lstm)
        while IFS= read -r line; do
            JOB_ENTRIES+=("$line")
        done < <(generate_replacements "lstm")
        ;;
    transformer)
        while IFS= read -r line; do
            JOB_ENTRIES+=("$line")
        done < <(generate_replacements "transformer")
        ;;
    gaussian)
        while IFS= read -r line; do
            JOB_ENTRIES+=("$line")
        done < <(generate_dist_replacements "gaussian")
        ;;
    kde)
        while IFS= read -r line; do
            JOB_ENTRIES+=("$line")
        done < <(generate_dist_replacements "kde")
        ;;
    *)
        echo "未知模式: $MODE"; exit 1 ;;
esac

TOTAL_JOBS=$(( ${#JOB_ENTRIES[@]} * ${#CV_LIST[@]} * ${#TASK_LIST[@]} ))

# ===================== 打印摘要 =====================
echo "================================================================"
echo "  ADFNet Ablation Study - Batch Parallel Launcher"
echo "================================================================"
echo "  模式:          $MODE"
echo "  消融配置数:    ${#JOB_ENTRIES[@]}"
echo "  交叉验证:      ${CV_LIST[*]}"
echo "  任务难度:      ${TASK_LIST[*]}"
echo "  总 Job 数:     $TOTAL_JOBS"
echo "  GPU:           ${GPU_ARRAY[*]:-CPU} (${NUM_GPUS} 张)"
echo "  每卡并行:      $MAX_PARALLEL_PER_GPU"
echo "  最大同时运行:  $MAX_JOBS"
echo "  输出目录:      $OUTPUT_DIR"
echo "  日志目录:      $LOG_DIR"
[[ -n "$MAX_FOLDS" ]] && echo "  Max folds:     $MAX_FOLDS (调试模式)"
echo "================================================================"
echo ""

# ===================== 提交所有 Job =====================
cd "$PROJECT_DIR"
mkdir -p "$LOG_DIR"

for entry in "${JOB_ENTRIES[@]}"; do
    abl_args="${entry%%|*}"
    label="${entry##*|}"
    # 去除末尾空格
    abl_args=$(echo "$abl_args" | sed 's/[[:space:]]*$//')
    launch_job "$label" "$abl_args"
done

# ===================== 等待全部完成 =====================
log "所有 Job 已提交, 等待完成..."
wait_all

echo ""
log "================================================================"
log "  全部消融实验完成!"
log "  结果目录: $OUTPUT_DIR"
log "  日志目录: $LOG_DIR"
log "================================================================"
