## Get Started
We have simple requirements in `requirements.txt`. You can always check if you can run the code immediately.

The datasets as well as the pretrained LM (LMsr) are uploaded here: hhttps://drive.google.com/drive/folders/1ifgVHQDnvFEunP9hmVYT07Y3rvcpIfQp?usp=sharing

Please download them and extract them to the corresponding folders.

## Training
Please follow the guidelines and hyperparamters of the corresponding GNNs for training. See `scripts` on a training example.  

Otherwise, you can download released GNN models from here: https://drive.google.com/file/d/1p7eLSsSKkZQxB32mT5lMsthVP6R_3x1j/view

## Evaluation

To evaluate them, copy the command from the above scripts, add the `--is_eval` argument, and `--load experiment` followed by the name of the corresponding `ckpt` model.

For example, for Webqsp run:
```
python main.py ReaRev --entity_dim 50 --num_epoch 200 --batch_size 8 --eval_every 2 --data_folder data/webqsp/ --lm sbert --num_iter 3 --num_ins 2 --num_gnn 3 --relation_word_emb True --load_experiment ReaRev_webqsp.ckpt --is_eval --name webqsp
```

The result is saved as a `.info` file. In order to use GNN-RAG, please move this file to the corresponding folder in `GNN-RAG/llm/results/gnn/` by renaming it to `test.info`.

## Subgraph Denoising (optional)
You can enable subgraph denoising during data loading to prune noisy edges:
```
python main.py ReaRev --data_folder data/webqsp/ --lm sbert --enable_denoise true --encoder_type tfidf --topM_strict 10 --topK_strict 50 --gamma_strict 0.2 --min_edges 50
```
If you have relation descriptions or relation-pair stats, provide them via:
```
--relation_desc_path path/to/relation_desc.json --enable_pair_score true --pair_stats_path path/to/pair_stats.json
```
Sanity check on one example:
```
python scripts/denoise_sanity.py --data_folder data/webqsp/ --split dev --sample_idx 0 --enable_denoise true
```

```mermaid
graph TD

    %% スタイル定義
    classDef loss fill:#f9d5e5,stroke:#333,stroke-width:2px;
    classDef embed fill:#e1f7d5,stroke:#333,stroke-width:2px;
    classDef core fill:#d5e8f7,stroke:#333,stroke-width:2px;
    classDef submod fill:#fff,stroke:#999,stroke-dasharray: 5 5;

    subgraph ReaRev["ReaRev Model (Total Params: ~113M)"]
        direction TB

        %% Loss Functions
        subgraph Losses["Loss Functions"]
            L1["KLDivLoss"]:::loss
            L2["BCEWithLogitsLoss"]:::loss
            L3["MSELoss"]:::loss
        end

        %% Embeddings & Projection
        subgraph Inputs["Embeddings & Initial Projection"]
            WE["Word Embedding<br/>(1, 768)"]:::embed
            RE["Relation Embedding<br/>(6651, 150)"]:::embed
            REI["Relation Embedding Inv<br/>(6651, 150)"]:::embed
            EL["Entity Linear<br/>100 → 150"]:::embed
            RL["Relation Linear<br/>150 → 150"]:::embed
        end

        %% Instruction Module
        subgraph Instruction["Instruction Module (BERTInstruction)"]
            direction TB
            BERT["BERT Model<br/>12 Layers / Hidden 768"]:::core
            QP["Question Projections<br/>Linear Layers 0-2"]:::submod
            BERT --> QP
        end

        %% Reasoning Module
        subgraph Reasoning["Reasoning Module (ReasonGNNLayer)"]
            direction TB

            GNN["GNN Core Logic"]:::core

            subgraph Steps["Multi-Step Reasoning"]
                S0["Step 0: Rel/E2E Linear"]:::submod
                S1["Step 1: Rel/E2E Linear"]:::submod
                S2["Step 2: Rel/E2E Linear"]:::submod
            end

            GNN --- S0
            GNN --- S1
            GNN --- S2
        end

        %% Query Reformulation
        subgraph Reform["Query Reformulation"]
            RF0["QueryReform 0"]:::core
            RF1["QueryReform 1"]:::core
            RF2["QueryReform 2"]:::core
            Fusion["Fusion Module"]:::submod
        end

        %% Connections
        Inputs --> Reasoning
        Instruction --> Reasoning
        Reasoning --> Reform
        Reform --> L2
    end
```

