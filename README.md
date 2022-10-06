# nlvr
Test
/vc_data/users/taoli1/nlvr$ cd nlvr2/
python util/download_images.py data/test1.json  ../../data/ util/hashes/test1_hashes.json 
python util/download_images.py data/dev.json  ../../data/ util/hashes/dev_hashes.json
python util/download_images.py data/train.json  ../../data/ util/hashes/train_hashes.json
cd ViLT && python nlvr2_arrow.py

python run.py with data_root=/vc_data/users/taoli1/data/nlvrarrow num_gpus=1  num_nodes=1 per_gpu_batchsize=64 task_finetune_nlvr2_randaug test_only=True precision=32 load_path="/vc_data/users/taoli1/mm/vilt/vilt_nlvr2.ckpt"

python ofa_eval_nlvr_cot_v1.py 0 1745 0
python ofa_eval_nlvr_cot_v1.py 1745 3940 1
python ofa_eval_nlvr_cot_v1.py 3940 5205 2
python ofa_eval_nlvr_cot_v1.py 5205 6967 3