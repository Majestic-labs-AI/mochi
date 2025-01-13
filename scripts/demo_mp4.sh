
# We set the resolution; which need to be a multiple of 16.
# Also, Instagram recommends 9:16 (portrait, not landscape)
# so the minimal width is 9 x 16 = 144, second minimal is
# (9 + 9) x 16 = 288, etc.
set -- \
    256 144 \
    512 288 \
    768 432 \
    1024 576 \
    1280 720

mochi_prompt="A rose budding then blossoming"
num_frames=97
while [ $# -gt 0 ]; do
    height=$1
    width=$2
    numactl --cpunodebind=0 --membind=0 python3 ./demos/cli.py \
        --model_dir weights/ \
        --cpu_offload \
        --num_frames $num_frames \
        --width $width \
        --height $height \
        --prompt "$mochi_prompt"
    shift 2
done