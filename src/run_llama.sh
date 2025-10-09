export CUDA_VISIBLE_DEVICES=0,1

python3 train_llm.py train_configs_llama/train_llama_0.json
python3 train_llm.py train_configs_llama/train_llama_1.json 
python3 train_llm.py train_configs_llama/train_llama_2.json 
python3 train_llm.py train_configs_llama/train_llama_3.json 
python3 train_llm.py train_configs_llama/train_llama_4.json 

python3 explain.py --config_file ../config_files/local_LLM/explain/ag_news/ag_news_std_dualda_0.1_v0.yaml --merge_explanations
python3 explain.py --config_file ../config_files/local_LLM/explain/ag_news/ag_news_std_dualda_0.1_v1.yaml --merge_explanations
python3 explain.py --config_file ../config_files/local_LLM/explain/ag_news/ag_news_std_dualda_0.1_v2.yaml --merge_explanations
python3 explain.py --config_file ../config_files/local_LLM/explain/ag_news/ag_news_std_dualda_0.1_v3.yaml --merge_explanations
python3 explain.py --config_file ../config_files/local_LLM/explain/ag_news/ag_news_std_dualda_0.1_v4.yaml --merge_explanations

python3 explain.py --config_file ../config_files/local_LLM/explain/ag_news/ag_news_std_dualda_0.001_v0.yaml --merge_explanations
python3 explain.py --config_file ../config_files/local_LLM/explain/ag_news/ag_news_std_dualda_0.001_v1.yaml --merge_explanations
python3 explain.py --config_file ../config_files/local_LLM/explain/ag_news/ag_news_std_dualda_0.001_v2.yaml --merge_explanations
python3 explain.py --config_file ../config_files/local_LLM/explain/ag_news/ag_news_std_dualda_0.001_v3.yaml --merge_explanations
python3 explain.py --config_file ../config_files/local_LLM/explain/ag_news/ag_news_std_dualda_0.001_v4.yaml --merge_explanations

python3 explain.py --config_file ../config_files/local_LLM/explain/ag_news/ag_news_std_dualda_1e-05_v0.yaml --merge_explanations
python3 explain.py --config_file ../config_files/local_LLM/explain/ag_news/ag_news_std_dualda_1e-05_v1.yaml --merge_explanations
python3 explain.py --config_file ../config_files/local_LLM/explain/ag_news/ag_news_std_dualda_1e-05_v2.yaml --merge_explanations
python3 explain.py --config_file ../config_files/local_LLM/explain/ag_news/ag_news_std_dualda_1e-05_v3.yaml --merge_explanations
python3 explain.py --config_file ../config_files/local_LLM/explain/ag_news/ag_news_std_dualda_1e-05_v4.yaml --merge_explanations

python3 train_llm.py train_configs_llama/train_llama_dualda_top_C-1_0.json
python3 train_llm.py train_configs_llama/train_llama_dualda_top_C-3_0.json
python3 train_llm.py train_configs_llama/train_llama_dualda_top_C-5_0.json

python3 train_llm.py train_configs_llama/train_llama_dualda_top_C-1_1.json
python3 train_llm.py train_configs_llama/train_llama_dualda_top_C-3_1.json
python3 train_llm.py train_configs_llama/train_llama_dualda_top_C-5_1.json

python3 train_llm.py train_configs_llama/train_llama_dualda_top_C-1_2.json
python3 train_llm.py train_configs_llama/train_llama_dualda_top_C-3_2.json
python3 train_llm.py train_configs_llama/train_llama_dualda_top_C-5_2.json

python3 train_llm.py train_configs_llama/train_llama_dualda_top_C-1_3.json
python3 train_llm.py train_configs_llama/train_llama_dualda_top_C-3_3.json
python3 train_llm.py train_configs_llama/train_llama_dualda_top_C-5_3.json

python3 train_llm.py train_configs_llama/train_llama_dualda_top_C-1_4.json
python3 train_llm.py train_configs_llama/train_llama_dualda_top_C-3_4.json
python3 train_llm.py train_configs_llama/train_llama_dualda_top_C-5_4.json