# train -> subgraph_train.sqlite
python3 merge_cwq_subgraphs.py --input data/CWQ/train.json --db data/CWQ/subgraph_train.sqlite --out data/CWQ/subgraph_train.json

# test -> subgraph_test.sqlite
python3 merge_cwq_subgraphs.py --input data/CWQ/test.json --db data/CWQ/subgraph_test.sqlite --out data/CWQ/subgraph_test.json

# dev -> subgraph_dev.sqlite（必要なら）
python3 merge_cwq_subgraphs.py --input data/CWQ/dev.json  --db data/CWQ/subgraph_dev.sqlite  --out data/CWQ/subgraph_dev.json
