set -e 
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
export MUJOCO_GL=osmesa

#python examples/libero/main.py
python examples/libero/main.py \
  --args.task-suite-name libero_10