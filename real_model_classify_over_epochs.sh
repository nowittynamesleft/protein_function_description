python train_and_test_model.py first_100_uniprot_training_annot.csv uniprot_sprot_training_val_split.csv --batch_size 1 --save_prefix real_model_generation_val_split_ep_0_ssl_32 --classify --num_pred_terms -1 --seq_set_len 32 --load_model_predict lightning_logs/real_model_training_split_experiment/version_0/checkpoints/epoch\=0-step\=9052.ckpt --load_vocab real_model_training_split_vocab.pckl > outfile_ep_0.txt
python train_and_test_model.py first_100_uniprot_training_annot.csv uniprot_sprot_training_val_split.csv --batch_size 1 --save_prefix real_model_generation_val_split_ep_10_ssl_32 --classify --num_pred_terms -1 --seq_set_len 32 --load_model_predict lightning_logs/real_model_training_split_experiment/version_0/checkpoints/epoch\=10-step\=99582.ckpt --load_vocab real_model_training_split_vocab.pckl > outfile_ep_10.txt
python train_and_test_model.py first_100_uniprot_training_annot.csv uniprot_sprot_training_val_split.csv --batch_size 1 --save_prefix real_model_generation_val_split_ep_20_ssl_32 --classify --num_pred_terms -1 --seq_set_len 32 --load_model_predict lightning_logs/real_model_training_split_experiment/version_0/checkpoints/epoch\=20-step\=190112.ckpt --load_vocab real_model_training_split_vocab.pckl > outfile_ep_20.txt
python train_and_test_model.py first_100_uniprot_training_annot.csv uniprot_sprot_training_val_split.csv --batch_size 1 --save_prefix real_model_generation_val_split_ep_30_ssl_32 --classify --num_pred_terms -1 --seq_set_len 32 --load_model_predict lightning_logs/real_model_training_split_experiment/version_0/checkpoints/epoch\=30-step\=280642.ckpt --load_vocab real_model_training_split_vocab.pckl > outfile_ep_30.txt
python train_and_test_model.py first_100_uniprot_training_annot.csv uniprot_sprot_training_val_split.csv --batch_size 1 --save_prefix real_model_generation_val_split_ep_40_ssl_32 --classify --num_pred_terms -1 --seq_set_len 32 --load_model_predict lightning_logs/real_model_training_split_experiment/version_0/checkpoints/epoch\=40-step\=371172.ckpt --load_vocab real_model_training_split_vocab.pckl > outfile_ep_40.txt
python train_and_test_model.py first_100_uniprot_training_annot.csv uniprot_sprot_training_val_split.csv --batch_size 1 --save_prefix real_model_generation_val_split_ep_50_ssl_32 --classify --num_pred_terms -1 --seq_set_len 32 --load_model_predict lightning_logs/real_model_training_split_experiment/version_0/checkpoints/epoch\=50-step\=461702.ckpt --load_vocab real_model_training_split_vocab.pckl > outfile_ep_50.txt 
python train_and_test_model.py first_100_uniprot_training_annot.csv uniprot_sprot_training_val_split.csv --batch_size 1 --save_prefix real_model_generation_val_split_ep_54_ssl_32 --classify --num_pred_terms -1 --seq_set_len 32 --load_model_predict lightning_logs/real_model_training_split_experiment/version_0/checkpoints/epoch\=54-step\=497914.ckpt --load_vocab real_model_training_split_vocab.pckl > outfile_ep_54.txt