# train tokenizer
CUDA_VISIBLE_DEVICES=1 python train_tokenizer.py \
    --data_dir 'data'\
    --output_dir 'tokenizer'\
    --lr 5e-4\
    --batch_size 128\
    --num_epochs 2\
    --contrastive 0


# get image tokens
python rq.py \
    --features 'tokenizer/tok_feats.npy' \
    --data_dir 'data'\
    --output_file 'codes.pkl' \
    --classes_output_file 'cls_codes.pkl'

# train irgen
CUDA_VISIBLE_DEVICES=1 python train_ar.py \
    --data_dir 'data'\
    --codes 'codes.pkl'\
    --cls_codes_file 'cls_codes.pkl' \
    --output_dir 'result_model'\
    --lr 4e-5 \
    --batch_size 32\
    --num_epochs 1\
    --num_workers 8\
    --smoothing 0.1\
    --contrastive 0

# test result
CUDA_VISIBLE_DEVICES=1 python valid_ar.py \
    --data_dir 'data'\
    --codes 'codes.pkl' \
    --model_dir 'result_model/ar.pkl' \
    --beam_size 30 \
    --ks 1 10 20 30\
    --output_dir 'result_pred'
