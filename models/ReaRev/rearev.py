import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

from models.base_model import BaseModel
from modules.kg_reasoning.reasongnn import ReasonGNNLayer
from modules.question_encoding.lstm_encoder import LSTMInstruction
from modules.question_encoding.bert_encoder import BERTInstruction
from modules.layer_init import TypeLayer
from modules.query_update import AttnEncoder, Fusion, QueryReform

VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000



class ReaRev(BaseModel):
    def __init__(self, args, num_entity, num_relation, num_word):
        """
        Init ReaRev model.
        """
        super(ReaRev, self).__init__(args, num_entity, num_relation, num_word)
        #self.embedding_def()
        #self.share_module_def()
        self.norm_rel = args['norm_rel']
        self.layers(args)
        

        self.loss_type =  args['loss_type']
        self.num_iter = args['num_iter']
        self.num_ins = args['num_ins']
        self.num_gnn = args['num_gnn']
        self.alg = args['alg']
        assert self.alg == 'bfs'
        self.lm = args['lm']
        
        self.private_module_def(args, num_entity, num_relation)

        self.to(self.device)
        self.lin = nn.Linear(3*self.entity_dim, self.entity_dim)

        self.fusion = Fusion(self.entity_dim)
        self.reforms = []
        for i in range(self.num_ins):
            self.add_module('reform' + str(i), QueryReform(self.entity_dim))
        # === 2段階推論のための設定 (ユーザー要求) ===
        # Pass 1 で使用する低い閾値
        self.refinement_threshold = args.get('refinement_threshold', 0.01) 
        # Pass 2 の絞り込み推論を実行するかどうかのフラグ
        self.run_refinement_pass = args.get('run_refinement_pass', True)
        # self.reform_rel = QueryReform(self.entity_dim)
        # self.add_module('reform', QueryReform(self.entity_dim))

    def layers(self, args):
        # initialize entity embedding
        word_dim = self.word_dim
        kg_dim = self.kg_dim
        entity_dim = self.entity_dim

        #self.lstm_dropout = args['lstm_dropout']
        self.linear_dropout = args['linear_dropout']
        
        self.entity_linear = nn.Linear(in_features=self.ent_dim, out_features=entity_dim)
        self.relation_linear = nn.Linear(in_features=self.rel_dim, out_features=entity_dim)
        # self.relation_linear_inv = nn.Linear(in_features=self.rel_dim, out_features=entity_dim)
        #self.relation_linear = nn.Linear(in_features=self.rel_dim, out_features=entity_dim)

        # dropout
        #self.lstm_drop = nn.Dropout(p=self.lstm_dropout)
        self.linear_drop = nn.Dropout(p=self.linear_dropout)

        if self.encode_type:
            self.type_layer = TypeLayer(in_features=entity_dim, out_features=entity_dim,
                                        linear_drop=self.linear_drop, device=self.device, norm_rel=self.norm_rel)

        self.self_att_r = AttnEncoder(self.entity_dim)
        #self.self_att_r_inv = AttnEncoder(self.entity_dim)
        self.kld_loss = nn.KLDivLoss(reduction='none')
        self.bce_loss_logits = nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = torch.nn.MSELoss()

    def get_ent_init(self, local_entity, kb_adj_mat, rel_features):
        if self.encode_type:
            local_entity_emb = self.type_layer(local_entity=local_entity,
                                               edge_list=kb_adj_mat,
                                               rel_features=rel_features)
        else:
            local_entity_emb = self.entity_embedding(local_entity)  # batch_size, max_local_entity, word_dim
            local_entity_emb = self.entity_linear(local_entity_emb)
        
        return local_entity_emb
    
   
    def get_rel_feature(self):
        """
        Encode relation tokens to vectors.
        """
        if self.rel_texts is None:
            rel_features = self.relation_embedding.weight
            rel_features_inv = self.relation_embedding_inv.weight
            rel_features = self.relation_linear(rel_features)
            rel_features_inv = self.relation_linear(rel_features_inv)
        else:
            
            rel_features = self.instruction.question_emb(self.rel_features)
            rel_features_inv = self.instruction.question_emb(self.rel_features_inv)
            
            rel_features = self.self_att_r(rel_features,  (self.rel_texts != self.instruction.pad_val).float())
            rel_features_inv = self.self_att_r(rel_features_inv,  (self.rel_texts != self.instruction.pad_val).float())
            if self.lm == 'lstm':
                rel_features = self.self_att_r(rel_features, (self.rel_texts != self.num_relation+1).float())
                rel_features_inv = self.self_att_r(rel_features_inv, (self.rel_texts_inv != self.num_relation+1).float())

        return rel_features, rel_features_inv


    def private_module_def(self, args, num_entity, num_relation):
        """
        Building modules: LM encoder, GNN, etc.
        """
        # initialize entity embedding
        word_dim = self.word_dim
        kg_dim = self.kg_dim
        entity_dim = self.entity_dim
        self.reasoning = ReasonGNNLayer(args, num_entity, num_relation, entity_dim, self.alg)
        if args['lm'] == 'lstm':
            self.instruction = LSTMInstruction(args, self.word_embedding, self.num_word)
            self.relation_linear = nn.Linear(in_features=entity_dim, out_features=entity_dim)
        else:
            self.instruction = BERTInstruction(args, self.word_embedding, self.num_word, args['lm'])
            #self.relation_linear = nn.Linear(in_features=self.instruction.word_dim, out_features=entity_dim)
        # self.relation_linear = nn.Linear(in_features=entity_dim, out_features=entity_dim)
        # self.relation_linear_inv = nn.Linear(in_features=entity_dim, out_features=entity_dim)

    def init_reason(self, curr_dist, local_entity, kb_adj_mat, q_input, query_entities):
        """
        Initializing Reasoning
        """
        # batch_size = local_entity.size(0)
        self.local_entity = local_entity
        self.instruction_list, self.attn_list = self.instruction(q_input)
        rel_features, rel_features_inv  = self.get_rel_feature()
        self.local_entity_emb = self.get_ent_init(local_entity, kb_adj_mat, rel_features)
        self.init_entity_emb = self.local_entity_emb
        self.curr_dist = curr_dist
        self.dist_history = []
        self.action_probs = []
        self.seed_entities = curr_dist
        
        self.reasoning.init_reason( 
                                   local_entity=local_entity,
                                   kb_adj_mat=kb_adj_mat,
                                   local_entity_emb=self.local_entity_emb,
                                   rel_features=rel_features,
                                   rel_features_inv=rel_features_inv,
                                   query_entities=query_entities)


    def calc_loss_label(self, curr_dist, teacher_dist, label_valid):
        tp_loss = self.get_loss(pred_dist=curr_dist, answer_dist=teacher_dist, reduction='none')
        tp_loss = tp_loss * label_valid
        cur_loss = torch.sum(tp_loss) / curr_dist.size(0)
        return cur_loss

    
    def forward(self, batch, training=False):
        """
        Forward function: creates instructions and performs GNN reasoning.
        (修正版: 2段階推論プロセスを実装)
        """

        # local_entity, query_entities, kb_adj_mat, query_text, seed_dist, answer_dist = batch
        local_entity, query_entities, kb_adj_mat, query_text, seed_dist, true_batch_id, answer_dist = batch
        local_entity = torch.from_numpy(local_entity).type('torch.LongTensor').to(self.device)
        # local_entity_mask = (local_entity != self.num_entity).float()
        query_entities = torch.from_numpy(query_entities).type('torch.FloatTensor').to(self.device)
        answer_dist = torch.from_numpy(answer_dist).type('torch.FloatTensor').to(self.device)
        seed_dist = torch.from_numpy(seed_dist).type('torch.FloatTensor').to(self.device)
        current_dist = Variable(seed_dist, requires_grad=True)

        q_input= torch.from_numpy(query_text).type('torch.LongTensor').to(self.device)
        
        if self.lm != 'lstm':
            pad_val = self.instruction.pad_val 
            query_mask = (q_input != pad_val).float()
        else:
            query_mask = (q_input != self.num_word).float()


        """
        Instruction generations (初期命令の生成)
        """
        self.init_reason(curr_dist=current_dist, local_entity=local_entity,
                         kb_adj_mat=kb_adj_mat, q_input=q_input, query_entities=query_entities)
        self.instruction.init_reason(q_input)
        for i in range(self.num_ins):
            relational_ins, attn_weight = self.instruction.get_instruction(self.instruction.relational_ins, step=i)
            self.instruction.instructions.append(relational_ins.unsqueeze(1))
            self.instruction.relational_ins = relational_ins
        
        self.dist_history.append(self.curr_dist)


        """
        =================================================================
        PASS 1: Global Search (ユーザーの要求 1)
        =================================================================
        """
        pass1_dist = current_dist
        
        for t in range(self.num_iter):
            relation_ins = torch.cat(self.instruction.instructions, dim=1)
            # GNN推論の入力分布を設定
            self.curr_dist = pass1_dist 
            
            # GNN Reasoning
            for j in range(self.num_gnn):
                # self.reasoning は self.curr_dist を内部で更新・使用
                self.curr_dist, global_rep = self.reasoning(self.curr_dist, relation_ins, step=j)
            
            pass1_dist = self.curr_dist # GNNの出力分布を更新

            """
            Instruction Updates (Pass 1)
            """
            qs = []
            for j in range(self.num_ins):
                reform = getattr(self, 'reform' + str(j))
                q = reform(self.instruction.instructions[j].squeeze(1), global_rep, query_entities, local_entity)
                qs.append(q.unsqueeze(1))
                self.instruction.instructions[j] = q.unsqueeze(1) # 命令を更新

        # Pass 1 の最終的な分布を履歴に保存
        self.dist_history.append(pass1_dist)
        pass1_updated_instructions = self.instruction.instructions

        """
        =================================================================
        PASS 1 の結果から候補を抽出し、Pass 2 の準備
        =================================================================
        """
        with torch.no_grad():
            # ユーザーが要求した「低い閾値」でマスクを作成
            candidate_mask = (pass1_dist > self.refinement_threshold).float()
            
            # 閾値で候補が0件の場合のフォールバック (例: Top-k)
            if torch.sum(candidate_mask) == 0:
                # print("Warning: No candidates found with threshold. Falling back to top-5.")
                _, top_k_indices = torch.topk(pass1_dist, k=5, dim=1)
                candidate_mask = torch.zeros_like(pass1_dist).scatter_(1, top_k_indices, 1.0)
        
        # マスクされた分布を作成 (これがPass 2の初期分布)
        refinement_start_dist = pass1_dist * candidate_mask
        
        # 確率の合計が0にならないようにepsilonを追加して再正規化
        refinement_start_dist = F.normalize(refinement_start_dist + VERY_SMALL_NUMBER, p=1, dim=1)
        
        # 勾配が流れるように Variable に変換
        refinement_start_dist = Variable(refinement_start_dist, requires_grad=True)

        
        """
        =================================================================
        PASS 2: Refinement (ユーザーの要求 2)
        =================================================================
        """
        if self.run_refinement_pass:
            # (重要) GNNの状態をリセット
            # init_reasonを再度呼び出し、GNNレイヤー内のエンティティ埋め込み(self.local_entity_emb)を
            # 初期状態に戻し、初期分布を絞り込んだもの(refinement_start_dist)に設定する。
            self.init_reason(curr_dist=refinement_start_dist, local_entity=local_entity,
                             kb_adj_mat=kb_adj_mat, q_input=q_input, query_entities=query_entities)
            
            # (注意) 命令(instruction)はPass 1で更新されたもの (self.instruction.instructions) が
            # init_reason を呼び出してもリセットされずに残っているため、更新済みの命令が使われる。
            self.instruction.instructions = pass1_updated_instructions
            
            refinement_dist = refinement_start_dist
            
            for t in range(self.num_iter): # 再度、同じ回数イテレーション
                # Pass 1 で更新済みの命令を使用
                relation_ins = torch.cat(self.instruction.instructions, dim=1)
                
                # GNN推論の入力分布を設定
                self.curr_dist = refinement_dist 
                
                for j in range(self.num_gnn):
                    self.curr_dist, global_rep = self.reasoning(self.curr_dist, relation_ins, step=j)
                
                refinement_dist = self.curr_dist # GNNの出力分布を更新

                # (オプション) Pass 2 でも命令を更新する
                qs = []
                for j in range(self.num_ins):
                    reform = getattr(self, 'reform' + str(j))
                    q = reform(self.instruction.instructions[j].squeeze(1), global_rep, query_entities, local_entity)
                    qs.append(q.unsqueeze(1))
                    self.instruction.instructions[j] = q.unsqueeze(1)

            final_pred_dist = refinement_dist
            self.dist_history.append(final_pred_dist)
        
        else:
            # Pass 2 を実行しない場合
            final_pred_dist = pass1_dist
            
        """
        =================================================================
        Answer Predictions
        =================================================================
        """
        pred_dist = final_pred_dist # self.dist_history[-1] を使う
        answer_number = torch.sum(answer_dist, dim=1, keepdim=True)
        case_valid = (answer_number > 0).float()

        # 損失計算
        # 最終的な予測(Pass 2)のみで損失を計算する
        loss = self.calc_loss_label(curr_dist=pred_dist, teacher_dist=answer_dist, label_valid=case_valid)

        # (オプション) Pass 1 の分布に対しても損失を加えたい場合:
        # loss_pass1 = self.calc_loss_label(curr_dist=pass1_dist, teacher_dist=answer_dist, label_valid=case_valid)
        # loss = 0.5 * loss_pass1 + 0.5 * loss # 重み付けして合計する

        pred = torch.max(pred_dist, dim=1)[1]
        if training:
            h1, f1 = self.get_eval_metric(pred_dist, answer_dist)
            tp_list = [h1.tolist(), f1.tolist()]
        else:
            tp_list = None
        return loss, pred, pred_dist, tp_list
    