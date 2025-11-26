
from tqdm import tqdm
tqdm.monitor_iterval = 0
import torch
import numpy as np
import math, os
import json
import pickle
import networkx as nx  # <-- 追加
import matplotlib.pyplot as plt # <-- 追加

def cal_accuracy(pred, answer_dist):
    """
    pred: batch_size
    answer_dist: batch_size, max_local_entity
    """
    num_correct = 0.0
    num_answerable = 0.0
    for i, l in enumerate(pred):
        num_correct += (answer_dist[i, l] != 0)
    for dist in answer_dist:
        if np.sum(dist) != 0:
            num_answerable += 1
    return num_correct / len(pred), num_answerable / len(pred)


def f1_and_hits(answers, candidate2prob, id2entity, entity2name, eps=0.5):
    ans = []
    retrieved = []
    for a in answers:
        if entity2name is None:
            ans.append(id2entity[a])
        else:
            ans.append(entity2name[id2entity[a]])
    correct = 0
    cand_list = sorted(candidate2prob, key=lambda x:x[1], reverse=True)
    if len(cand_list) == 0:
        best_ans = -1
    else:
        best_ans = cand_list[0][0]
    # max_prob = cand_list[0][1]
    tp_prob = 0.0
    for c, prob in cand_list:
        if entity2name is None:
            retrieved.append((id2entity[c], prob))
        else:
           retrieved.append((entity2name[id2entity[c]], prob))
        tp_prob += prob
        if c in answers:
            correct += 1
        if tp_prob > eps:
            break
    if correct > 0:
        em = 1
    else:
        em = 0
    if len(answers) == 0:
        if len(retrieved) == 0:
            return 1.0, 1.0, 1.0, 1.0, 1.0, 0, retrieved, ans  # precision, recall, f1, hits, em
        else:
            return 0.0, 1.0, 0.0, 1.0, 1.0, 1, retrieved , ans # precision, recall, f1, hits, em
    else:
        hits = float(best_ans in answers)
        if len(retrieved) == 0:
            return 1.0, 0.0, 0.0, hits, hits, 2, retrieved , ans # precision, recall, f1, hits, em
        else:
            p, r = correct / len(retrieved), correct / len(answers)
            f1 = 2.0 / (1.0 / p + 1.0 / r) if p != 0 and r != 0 else 0.0
            return p, r, f1, hits, em, 3, retrieved, ans


class Evaluator:

    def __init__(self, args, model, entity2id, relation2id, device):
        self.model = model
        self.args = args
        self.eps = args['eps']
        self.model_name = args["model_name"]
        
        id2entity = {idx: entity for entity, idx in entity2id.items()}
        self.id2entity = id2entity

        self.entity2name = None
        if 'sr-' in args["data_folder"]:
            file = open('ent2id.pickle', 'rb')
            self.entity2name = list((pickle.load(file)).keys())
            file.close()

            
        id2relation = {idx: relation for relation, idx in relation2id.items()}
        num_rel_ori = len(relation2id)

        if 'use_inverse_relation' in args:
            self.use_inverse_relation = args['use_inverse_relation']
            if self.use_inverse_relation:
                for i in range(len(id2relation)):
                    id2relation[i + num_rel_ori] = id2relation[i] + "_rev"

        if 'use_self_loop' in args:
            self.use_self_loop = args['use_self_loop']
            if self.use_self_loop:
                id2relation[len(id2relation)] = "self_loop"

        self.id2relation = id2relation
        self.file_write = None
        self.device = device

    def write_info(self, valid_data, tp_list, num_step):
        question_list = valid_data.get_quest()
        #num_step = steps
        obj_list = []
        if tp_list is not None:
            # attn_list = [tp[1] for tp in tp_list]
            action_list = [tp[0] for tp in tp_list]
        for i in range(len(question_list)):
            obj_list.append({})
        for j in range(num_step):
            if tp_list is None:
                actions = None
            else:
                actions = action_list[j]
                actions = actions.cpu().numpy()
            # if attn_list is not None:
            #     attention = attn_list[j].cpu().numpy()
            for i in range(len(question_list)):
                tp_obj = obj_list[i]
                q = question_list[i]
                # real_index = self.true_batch_id[i][0]
                tp_obj['question'] = q
                tp_obj[j] = {}
                # print(actions)
                if tp_list is not None:
                    action = actions[i]
                    rel_action = self.id2relation[action]
                    tp_obj[j]['rel_action'] = rel_action
                    tp_obj[j]['action'] = str(action)
                    # if attn_list is not None:
                    #     attention_tp = attention[i]
                    #     tp_obj[j]['attention'] = attention_tp.tolist()
        return obj_list

    def evaluate(self, valid_data, test_batch_size=20, write_info=False):
        write_info = True
        self.model.eval()
        self.count = 0
        eps = self.eps
        id2entity = self.id2entity
        eval_loss, eval_acc, eval_max_acc = [], [], []
        f1s, hits, ems,  precisions, recalls = [], [], [], [], []
        valid_data.reset_batches(is_sequential=True)
        num_epoch = math.ceil(valid_data.num_data / test_batch_size)
        if write_info and self.file_write is None:
            filename = os.path.join(self.args['checkpoint_dir'],
                                    "{}_test.info".format(self.args['experiment_name']))
            self.file_write = open(filename, "w")
        case_ct = {}
        max_local_entity = valid_data.max_local_entity
        ignore_prob = (1 - eps) / max_local_entity
        for iteration in tqdm(range(num_epoch)):
            start = test_batch_size * iteration
            end = min(test_batch_size * (iteration + 1), valid_data.num_data)

            batch = valid_data.get_batch(iteration, test_batch_size, fact_dropout=0.0, test=True)
            with torch.no_grad():
                # loss, extras, pred_dist, tp_list = self.model(batch[:-1])
                # pred = torch.max(pred_dist, dim=1)[1]
                # === 修正 (デバッグ情報を受け取る) ===
                if self.model_name == 'GraftNet':
                    model_outputs = self.model(batch[:-1])
                else:
                    model_outputs = self.model(batch) # ReaRev は batch をそのまま渡す
                
                loss, extras, pred_dist, tp_list = model_outputs[0], model_outputs[1], model_outputs[2], model_outputs[3]
                
                debug_info = None
                if len(model_outputs) > 4:
                    debug_info = model_outputs[4]
                    # === 追加: 閾値調整のための確率分布表示 ===
            # if debug_info is not None and "pass1_dist" in debug_info:
            #     pass1_dists = debug_info["pass1_dist"] # shape: (Batch, MaxLocal)
                
            #     # バッチの最初のサンプルの統計を表示
            #     # (全てのサンプルを表示するとログが流れすぎるため)
            #     sample_probs = pass1_dists[0]
                
            #     # 統計量
            #     max_p = np.max(sample_probs)
            #     mean_p = np.mean(sample_probs)
            #     median_p = np.median(sample_probs)
                
            #     # 上位10個の確率値を表示
            #     # これを見ることで「有効な候補」と「ノイズ」の境界（閾値）の目安がつきます
            #     top_10_probs = np.sort(sample_probs)[-10:][::-1]
                
            #     print(f"\n--- [Batch {iteration}] Pass 1 Statistics (Sample 0) ---")
            #     print(f"  Max: {max_p:.6f}, Mean: {mean_p:.6f}, Median: {median_p:.6f}")
            #     print(f"  Top 10 Probs: {top_10_probs}")
            #     print("----------------------------------------------------")
            # # ===========================================
                # ====================================
            if self.model_name == 'GraftNet':
                local_entity, query_entities, _, _, query_text, _, \
                seed_dist, true_batch_id, answer_dist, answer_list = batch
            else:
                local_entity, query_entities, _, query_text, \
                seed_dist, true_batch_id, answer_dist, answer_list = batch
            # self.true_batch_id = true_batch_id
            batch_size = pred_dist.size(0)
            if write_info:
                obj_list = self.write_info(valid_data, tp_list, self.model.num_iter)
                # pred_sum = torch.sum(pred_dist, dim=1)
                # print(pred_sum)

                # === 追加: サブグラフのノード情報を抽出して保存 ===
                if debug_info is not None and "refinement_start_dist" in debug_info:
                    refinement_dists = debug_info["refinement_start_dist"] # (Batch, MaxLocal)
                    
                    # バッチ内の各サンプルについて処理
                    for batch_id in range(batch_size):
                        sample_id = valid_data.batches[start + batch_id]
                        
                        # ローカルID -> グローバルID のマッピングを取得
                        g2l = valid_data.global2local_entity_maps[sample_id]
                        l2g = {v: k for k, v in g2l.items()} # 逆引きマップ
                        
                        # 確率が0より大きい（サブグラフに含まれる）ローカルノードIDを取得
                        dist = refinement_dists[batch_id]
                        candidate_local_indices = np.where(dist > 0)[0]
                        
                        subgraph_node_names = []
                        for local_idx in candidate_local_indices:
                            if local_idx in l2g:
                                global_id = l2g[local_idx]
                                
                                # グローバルIDからエンティティ名(またはKB ID)に変換
                                node_label = str(global_id)
                                if global_id in id2entity:
                                    kb_id = id2entity[global_id]
                                    if self.entity2name is not None:
                                        try:
                                            if isinstance(self.entity2name, dict) and kb_id in self.entity2name:
                                                node_label = self.entity2name[kb_id]
                                            elif isinstance(self.entity2name, list) and int(kb_id) < len(self.entity2name):
                                                node_label = self.entity2name[int(kb_id)]
                                            else:
                                                node_label = str(kb_id)
                                        except:
                                            node_label = str(kb_id)
                                    else:
                                        node_label = str(kb_id)
                                
                                subgraph_node_names.append(node_label)
                        
                        # 結果オブジェクトに追加
                        obj_list[batch_id]['subgraph_nodes'] = subgraph_node_names
                # =================================================


            candidate_entities = torch.from_numpy(local_entity).type('torch.LongTensor')
            true_answers = torch.from_numpy(answer_dist).type('torch.FloatTensor')
            query_entities = torch.from_numpy(query_entities).type('torch.LongTensor')
            # acc, max_acc = cal_accuracy(pred, true_answers.cpu().numpy())
            eval_loss.append(loss.item())
            # eval_acc.append(acc)
            # eval_max_acc.append(max_acc)
            #pr_dist2 = pred_dist#.copy()
            #pred_dist = pr_dist2[-1]
            batch_size = pred_dist.size(0)
            batch_answers = answer_list
            batch_candidates = candidate_entities
            pad_ent_id = len(id2entity)
            #pr_dist2 = pred_dist.copy()
            #for pred_dist in pr_dist2:
            
            if debug_info is not None and iteration == 0 and write_info: 
                print("Generating subgraph visualization...")
                
                # 修正: start と end をここで定義する
                end = min(test_batch_size * (iteration + 1), valid_data.num_data)
                
                self.visualize_subgraph(
                    debug_info, 
                    candidate_entities, 
                    valid_data.global2local_entity_maps, 
                    valid_data.batches[start: end], 
                    self.id2entity,
                    self.entity2name
                )
            
            for batch_id in range(batch_size):
                answers = batch_answers[batch_id]
                candidates = batch_candidates[batch_id, :].tolist()
                probs = pred_dist[batch_id, :].tolist()
                seed_entities = query_entities[batch_id, :].tolist()
                #print(seed_entities)
                #print(candidates)
                candidate2prob = []
                for c, p, s in zip(candidates, probs, seed_entities):
                    if s == 1.0:
                        # ignore seed entities
                        #print(c, self.id2entity)
                        # print(c, p, s)
                        # if c < pad_ent_id:
                        #     tp_obj['seed'] = self.id2entity[c]
                        continue
                    if c == pad_ent_id:
                        continue
                    if p < ignore_prob:
                        continue
                    candidate2prob.append((c, p))
                precision, recall, f1, hit, em, case, retrived , ans = f1_and_hits(answers, candidate2prob, self.id2entity, self.entity2name ,eps)
                if write_info:
                    tp_obj = obj_list[batch_id]
                    tp_obj['answers'] = ans
                    tp_obj['precison'] = precision
                    tp_obj['recall'] = recall
                    tp_obj['f1'] = f1
                    tp_obj['hit'] = hit
                    tp_obj['em'] = em
                    tp_obj['cand'] = retrived
                    self.file_write.write(json.dumps(tp_obj) + "\n")
                case_ct.setdefault(case, 0)
                case_ct[case] += 1
                f1s.append(f1)
                hits.append(hit)
                ems.append(em)
                precisions.append(precision)
                recalls.append(recall)
        print('evaluation.......')
        print('how many eval samples......', len(f1s))
        # print('avg_f1', np.mean(f1s))
        print('avg_em', np.mean(ems))
        print('avg_hits', np.mean(hits))
        print('avg_f1', np.mean(f1s))
        print('avg_precision', np.mean(precisions))
        print('avg_recall', np.mean(recalls))
        
        print(case_ct)
        if write_info:
            self.file_write.close()
            self.file_write = None
        return np.mean(f1s), np.mean(hits), np.mean(ems)

    # (Evaluator クラスの evaluate メソッドの直後などに追加)
    def visualize_subgraph(self, debug_info, local_entity_batch, global2local_maps_all, sample_ids, id2entity, entity2name, batch_id=0):
        """
        デバッグ情報からサブグラフを可視化し、保存する。
        """
        try:
            # --- 1. 必要な情報をバッチから抽出 (batch_id=0 のみ) ---
            sample_id = sample_ids[batch_id]
            g2l = global2local_maps_all[sample_id]
            # l2g (ローカルID -> グローバルID) の逆引きマップを作成
            l2g = {l: g for g, l in g2l.items()}

            local_entities = local_entity_batch[batch_id] # (max_local_entity,)
            pass1_dist = debug_info["pass1_dist"][batch_id] # (max_local_entity,)
            refinement_dist = debug_info["refinement_start_dist"][batch_id] # (max_local_entity,)
            seed_nodes_mask = debug_info["query_entities"][batch_id] # (max_local_entity,)
            
            # kb_adj_mat (batch_heads, batch_rels, batch_tails, batch_ids, ...)
            kb_adj_mat = debug_info["kb_adj_mat"]
            batch_heads, batch_rels, batch_tails, batch_ids = kb_adj_mat[0], kb_adj_mat[1], kb_adj_mat[2], kb_adj_mat[3]
            
            # --- 2. NetworkX グラフの構築 ---
            G = nx.DiGraph()
            
            node_labels = {}
            node_sizes_map = {} # ノードIDをキーにしたサイズマップ
            node_colors_map = {} # ノードIDをキーにしたカラーマップ
            
            valid_nodes = set() # 実際にグラフに存在するノード（Paddingを除く）

            # ノードを追加
            for local_id, global_id in enumerate(local_entities):
                if global_id == len(id2entity): # Padding ID
                    continue 
                
                valid_nodes.add(local_id)
                
                # ラベルの決定
                label = f"L:{local_id}\nG:{global_id}"
                if global_id in id2entity:
                    entity_mid = id2entity[global_id]
                    if entity2name is not None:
                        # entity2name がリストか辞書か不明なため、両対応を試みる
                        try:
                            if isinstance(entity2name, list) and int(entity_mid) < len(entity2name):
                                label = entity2name[int(entity_mid)]
                            elif isinstance(entity2name, dict) and entity_mid in entity2name:
                                label = entity2name[entity_mid]
                            else:
                                label = str(entity_mid)
                        except:
                             label = str(entity_mid) # フォールバック
                    else:
                        label = str(entity_mid)
                
                node_labels[local_id] = label
                
                # サイズと色の決定 (refinement_dist に基づく)
                prob = refinement_dist[local_id]
                size = 100 + prob * 5000 # 確率に応じてサイズ変更
                
                color = 'skyblue' # デフォルト (確率 0)
                if seed_nodes_mask[local_id] > 0:
                    color = 'red' # シードノード
                elif prob > 0:
                    color = 'orange' # Pass 1 候補ノード
                
                G.add_node(local_id)
                node_sizes_map[local_id] = size
                node_colors_map[local_id] = color

            # エッジを追加 (バッチ全体から該当バッチIDのタプルを抽出)
            # max_local_entity へのアクセスを self.model.reasoning から行う
            max_local_entity = self.model.reasoning.max_local_entity
            index_bias = batch_id * max_local_entity
            
            for h, r, t, b_id in zip(batch_heads, batch_rels, batch_tails, batch_ids):
                if b_id != batch_id:
                    continue
                
                # バッチ内ローカルIDに変換
                local_h = h - index_bias
                local_t = t - index_bias
                
                # 有効なノード（Paddingでない）間のエッジのみ追加
                if local_h in valid_nodes and local_t in valid_nodes:
                    G.add_edge(local_h, local_t, label=self.id2relation.get(r, str(r)))

            # --- 3. 描画と保存 ---
            if not G.nodes():
                print(f"Skipping visualization for sample {sample_id}: No valid nodes.")
                return
            
            # G.nodes() に基づいて色とサイズリストを再構築
            node_list = sorted(list(G.nodes())) # ノードの順序を固定
            node_sizes = [node_sizes_map[node_id] for node_id in node_list]
            node_colors = [node_colors_map[node_id] for node_id in node_list]
            filtered_labels = {n: node_labels[n] for n in node_list}


            plt.figure(figsize=(24, 18))
            pos = nx.spring_layout(G, k=0.8, iterations=50) # レイアウト
            
            # ノード描画
            nx.draw_networkx_nodes(G, pos, nodelist=node_list, node_size=node_sizes, node_color=node_colors, alpha=0.8)
            
            # エッジ描画
            nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=10, alpha=0.5, node_size=node_sizes)
            
            # ラベル描画
            nx.draw_networkx_labels(G, pos, labels=filtered_labels, font_size=8)
            
            # エッジラベル描画
            edge_labels = nx.get_edge_attributes(G, 'label')
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6, alpha=0.7)
            
            plt.title(f"Subgraph Visualization (Sample ID: {sample_id}) - Nodes sized by refinement_dist\nRed=Seed, Orange=Candidate (Prob>0), Blue=Ignored (Prob=0)")
            plt.axis('off')
            
            # 保存
            output_dir = self.args['checkpoint_dir']
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            filename = os.path.join(output_dir, f"{self.args['experiment_name']}_subgraph_sample_{sample_id}.png")
            plt.savefig(filename)
            plt.close()
            print(f"Subgraph visualization saved to {filename}")

        except Exception as e:
            print(f"Error during visualization: {e}")
            import traceback
            traceback.print_exc()

