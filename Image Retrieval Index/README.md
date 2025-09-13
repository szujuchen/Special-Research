#  Image GenIR

## 0. Environment
`conda env create -f environment.yaml`

## 1. Dataset
Put images into `{data_dir}/images/` in `train/` or `valid/`. 
Same classes images would be put in the same subfolder.
Each dataset should be in different `{data_dir}`.

```
-- images/ -- train/ -- 01 RedBird/   -- xxx.jpg
                                      -- xxx.jpg
                     -- 02 BlackBird/ -- xxx.jpg
                                      -- xxx.jpg
```

These two scripts would generated ground truth pickle file in {data_dir} for training.
```
python label_generator.py 
python gnd_generator.py

optional argument:
--data_dir          directory with trained images for different dataset
```

## 2. Tokenizer
The every 50 epochs and last epoch tokenizer would be saved in `output_dir` (`{data_dir}_model`).
```
python train_tokenizer.py 

optional arguments:
--data_dir          directory that contains the label and ground truth files [default 'data']
--feats             initialize features for quantize [default '']
--output_dir        directory save the tokenizer output model [default '{data_dir}_model']
--lr                learning rate [default 5e-4]
--batch_size        training batch size [default 128]
--num_epochs        training epochs [default 200]
--num_workers       training data loader worker [default 8]
--id_len            image token length [default 4]
--codebook_size     codebook size [default 8 (2^8)]
--contrastive       percentage of contrastive loss [range 0 - 1][default 0]
```

## 3. Image Code Token
```
python rq.py

optional arguments:
--data_dir              directory where save the generated tokens and load ground truth file [default 'data']
--features              trained tokenizer features files [default '{data_dir}_model/tok_feats.npy']
--output_file           file name of generated codes [default 'codes.pkl']
--classes_output_file   file name of the generated codes tree for different classes [default 'cls_codes.pkl']
--id_len                image token size (should be same as step2) [default 4]
--codebook_size         codebook size (shoule be same as step2) [default 8]
```

## 4. AutoRegressive
The every 50 epochs and last epoch model would be saved in `output_dir` (`{data_dir}_model`).
```
python train_ar.py
(use torch distributed python -m torch.distributed.launch --nproc_per_node=2 train_ar.py for multi gpu)

optional arguments:
--data_dir          directory where save the generated tokens and load ground truth file [default 'data']
--codes             file name of the generated codes in step3 [default 'codes.pkl']
--cls_codes_file    file name of the generated codes tree for different classes in step3 [default 'cls_codes.pkl']
--output_dir        directory save the autoregressive output model [default '{data_dir}_model']
--lr                learning rate [default 4e-5]
--batch_szie        training batch size [default 64]
--num_epochs        training epochs [default 200]
--num_workers       train data loader worker [default 8]
--smoothing         label smoothing in cross entropy [default 0.1]
--contrastive       percentage to use contrastive loss [default 1]
```

## 5. Validation (Images dataloader)
This step is for validation (images in the `{data_dir}/images/valid`)
The validation data would be save in the `{data_dir}_valid`.
```
python valid_ar.py 

optional argumment:
--data_dir          directory where save the generated tokens and load ground truth file [default 'data']
--codes             file name of the generated codes in step3 [default 'codes.pkl']
--model_dir         trained autoregressive model path [default '{data_dir}_model/ar.pkl']
--beam_size         number of predictions [default 30]
--ks                list to calculate metric [default 1 10 20 30] (should smaller than beam size)
--output_dir        directory where save the validation results [default '{data_dir}_valid']
```

Visualize the validation results
```
python getvalidation.py

optional argument:
--data_dir          directory where save the generated tokens and load ground truth file [default 'data'] 
--pred_dir          directory where save the validation results [default '{data_dir}_valid']
--size              visualization number of images [default 5]
```

## 6. Test (Images path)
The predictions images would be saved in the `{data_dir}_pred`.
```
python predict.py --query_img {search_image_path}

optional arguments:
--data_dir          directory where save the generated tokens and load ground truth file [default 'data']
--codes             file name of the generated codes in step3 [default 'codes.pkl']
--model_dir         trained autoregressive model path [default '{data_dir}_model/ar.pkl']
--beam_size         number of predictions [default 5]
--output_dir        directory where save the prediction images [default '{data_dir}_pred']
```

