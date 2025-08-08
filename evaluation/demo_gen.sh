python sample_c2i.py \
--image-size 384 \
--downsample-size 16 \
--gpt-model GPT-XXL \
--vq-model mgvq-f16c32 \
--codebook-size 32768 \
--codebook-groups 4 \
--codebook-embed-dim 32 \
--groups-to-use 4 \
--top-k 12 \
--cfg-scale 4.0 \
--gpt-ckpt /horizon-bucket/saturn_v_dev/mingkai.jia/pretrained_models/dcvq/tmp/arhead_328ep.pt \
--vq-ckpt /horizon-bucket/saturn_v_dev/mingkai.jia/pretrained_models/dcvq/tmp/in_f16_c2f_useall_540k.pt
# --gpt-ckpt \path\to\your\gpt_ckpt
# --vq-ckpt \path\to\your\vq_ckpt
