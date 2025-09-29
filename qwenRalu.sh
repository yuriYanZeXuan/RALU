python RALU_inference.py \
    --model_type qwen \
    --prompt "add a pair of sunglasses to the toy" \
    --edit_image_path "/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/imgs/glass_toy.jpg" \
    --output_dir "/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/imgs" \
    --flux_model_path "/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/flux" \
    --qwen_edit_path "/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/QwenEdit" \
    --use_ralu \
    --level 4