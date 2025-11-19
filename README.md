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


```mermaid
flowchart TD

    subgraph ReaRev[ReaRev Model]
        direction TB

        %% Embeddings
        subgraph EMB[Embeddings]
            WORD_EMB[Word Embedding\n(1 → 768)]
            REL_EMB[Relation Embedding\n6651 → 150]
            REL_EMB_INV[Relation Embedding Inv\n6651 → 150]
        end

        %% Linear Layers
        subgraph LINEAR[Linear Transforms]
            ENT_LIN[entity_linear\n100 → 150]
            REL_LIN[relation_linear\n150 → 150]
            LIN_DROP[Dropout 0.2]
        end

        %% Type Layer
        subgraph TYPE[TypeLayer]
            TYPE_DROP[Dropout 0.2]
            TYPE_KB[kb_self_linear\n150 → 150]
        end

        %% Self Attention
        subgraph SELFATT[SelfAtt (AttnEncoder)]
            KEY[key_linear\n150 → 150]
            VAL[value_linear\n150 → 150]
        end

        %% Reasoning GNN
        subgraph GNN[ReasonGNNLayer]
            SOFTMAX[Softmax dim=1]
            SCORE[score_func\n150 → 1]
            GLOB[glob_lin\n150 → 150]
            LIN_GNN[lin\n300 → 150]
            REL0[rel_linear0\n150 → 150]
            E2E0[e2e_linear0\n1050 → 150]
            REL1[rel_linear1\n150 → 150]
            E2E1[e2e_linear1\n1050 → 150]
            REL2[rel_linear2\n150 → 150]
            E2E2[e2e_linear2\n1050 → 150]
            LIN_M[lin_m\n450 → 150]
        end

        %% Instruction Module
        subgraph INST[BERTInstruction]
            INST_DROP[Dropout]
            WORD_EMB_I[word_embedding 1→768]
            CQ[cq_linear 600→150]
            CA[ca_linear 150→1]
            QL0[question_linear0 150→150]
            QL1[question_linear1 150→150]
            QL2[question_linear2 150→150]
            Q_EMB[question_emb 768→150]

            subgraph BERT[BERT Encoder]
                BERT_EMB[BertEmbeddings]
                BERT_ENC[BertEncoder (12 layers)]
                BERT_POOL[BertPooler]
            end
        end

        %% Fusion
        subgraph FUS[Fusion]
            FUS_R[r linear\n450→150]
            FUS_G[g linear\n450→150]
        end

        %% Reform Layers
        subgraph REFORM[QueryReform ×3]
            REFORM0[Reform0]
            REFORM1[Reform1]
            REFORM2[Reform2]
        end
    end

    %% Connections
    WORD_EMB --> ENT_LIN
    REL_EMB --> REL_LIN
    ENT_LIN --> TYPE
    REL_LIN --> SELFATT
    SELFATT --> GNN
    GNN --> FUS
    INST --> FUS
    FUS --> REFORM
```
