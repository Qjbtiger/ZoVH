TASK=${TASK:-SST2}
OPT_NAME=${OPT_NAME:-zovh} # hizoo, mezo, zovh
NUM_HISTORIES=${NUM_HISTORIES:-0}
ZoVH_LAMBDA_HESS=${ZoVH_LAMBDA_HESS:-0.5}
HESS_SCHEDULER_STEP=${HESS_SCHEDULER_STEP:-1500}
ZOAR_SCHEDULER_STEP=${ZOAR_SCHEDULER_STEP:-200}

MODEL=${MODEL:-facebook/opt-1.3b}
MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"

BS=${BS:-16}
LR=${LR:-5e-5}
EPS=${EPS:-1e-2}
SEED=${SEED:-0}
TRAIN=${TRAIN:-1000}
DEV=${DEV:-500}
EVAL=${EVAL:-1000}
STEPS=${STEPS:-5000}
EVAL_STEPS=${EVAL_STEPS:-2000}
WARMUP_STEP=${WARMUP_STEP:-0}
DECAY_STEP=${DECAY_STEP:-0}
ZO_LR_SCHEDULER_TYPE=${ZO_LR_SCHEDULER_TYPE:-'constant'}
WEIGHT_DECAY=${WEIGHT_DECAY:-0}
HESSIAN_SMOOTH_TYPE=${HESSIAN_SMOOTH_TYPE:-'constant1e-8'}
MODE=${MODE:-lora}
EXTRA_ARGS=""
if [ "$MODE" == "prefix" ]; then
    EXTRA_ARGS="--prefix_tuning --num_prefix 5 --no_reparam --prefix_init_by_real_act"
elif [ "$MODE" == "lora" ]; then
    EXTRA_ARGS="--lora"
fi
TAG=$OPT_NAME-$MODE-$STEPS-$BS-$LR-$EPS-$NUM_HISTORIES-$ZoVH_LAMBDA_HESS-$SEED-hesssched-${HESS_SCHEDULER_STEP}-zoarsched-${ZOAR_SCHEDULER_STEP}

TASK_ARGS=""
case $TASK in
    # For Copa, ReCoRD, SQuAD, DROP, we set --train_as_classification False; for others, set this flag to True
    CB) # It has <1000 training examples. Only use 100 for dev
        DEV=100
        ;;
    Copa) # It has <1000 training examples. Only use 100 for dev
        DEV=100
        TASK_ARGS="--train_as_classification False"
        ;;
    ReCoRD) 
        TASK_ARGS="--train_as_classification False"
        ;;
    DROP) 
        TASK_ARGS="--train_as_classification False"
        ;;
    SQuAD)
        TASK_ARGS="--train_as_classification False"
        ;;
esac

echo $TAG
echo "BS: $BS"
echo "LR: $LR"
echo "EPS: $EPS"
echo "SEED: $SEED"
echo "TRAIN/EVAL STEPS: $STEPS/$EVAL_STEPS"
echo "MODE: $MODE"
echo "Extra args: $EXTRA_ARGS $TASK_ARGS"
echo "OPT_NAME: $OPT_NAME"
echo "NUM_HISTORIES: $NUM_HISTORIES"
echo "ZoVH_LAMBDA_HESS: $ZoVH_LAMBDA_HESS"

WANDB_PROJECT=HessApprox python run.py \
    --opt_name $OPT_NAME \
    --num_histories $NUM_HISTORIES \
    --zovh_lambda_hess $ZoVH_LAMBDA_HESS \
    --hess_scheduler_step $HESS_SCHEDULER_STEP \
    --zoar_scheduler_step $ZOAR_SCHEDULER_STEP \
    --model_name $MODEL \
    --task_name $TASK \
    --output_dir result/$TASK-${MODEL_NAME}-$TAG --tag $TAG --train_set_seed $SEED --num_train $TRAIN --num_dev $DEV --num_eval $EVAL \
    --report_to wandb --logging_steps 1 --logging_strategy steps --logging_first_step true \
    --max_steps $STEPS \
    --trainer zo --load_float16 \
    --learning_rate $LR --zo_eps $EPS --per_device_train_batch_size $BS --lr_scheduler_type "constant" \
    --load_best_model_at_end --evaluation_strategy steps --save_strategy steps --save_total_limit 1 \
    --eval_steps $EVAL_STEPS --save_steps $EVAL_STEPS \
    --warmup_step $WARMUP_STEP --decay_step $DECAY_STEP  --zo_lr_scheduler_type $ZO_LR_SCHEDULER_TYPE \
    --weight_decay $WEIGHT_DECAY --hessian_smooth_type $HESSIAN_SMOOTH_TYPE \
    --train_as_classification \
    $EXTRA_ARGS \
    $TASK_ARGS \
    "$@"
