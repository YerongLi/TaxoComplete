import networkx as nx
import math
import argparse
import torch
from torch.utils.data import DataLoader
import data_process.split_data as st
import data_process.data_loader as dl
import logging
from model.sbert import SentenceTransformer, losses
from model.sbert.evaluation import EmbeddingSimilarityEvaluator
import compute_metrics.metric as ms
from parse_config import ConfigParser
from model.utils import PPRPowerIteration
import os
import pickle
def print_outgoing_edges(graph, node):
    for edge in graph.out_edges(node):
        print(edge)

def find_paths_of_length(graph, start_node, depth=2):
    paths = []

    def dfs(current_node, path, length):
        if length == depth:
            paths.append(path + [current_node])
            return

        for neighbor in graph.neighbors(current_node):
            if neighbor not in path:
                dfs(neighbor, path + [current_node], length + 1)

    dfs(start_node, [], 0)
    return paths

logging.basicConfig(
    format='%(asctime)s %(levelname)-4s - %(filename)-6s:%(lineno)d - %(message)s',
    level=logging.INFO,
    filename='./output.log',
    datefmt='%m-%d %H:%M:%S')
logging.info(f'Logger start: {os.uname()[1]}')

torch.manual_seed(0)
args = argparse.ArgumentParser(description='Training taxonomy expansion model')
args.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
config = ConfigParser(args)
args = args.parse_args()

saving_path = config['saving_path']
name = config['name']
data_path = config['data_path']
sampling_method = config['sampling']
neg_number = config['neg_number']
partition_pattern = config['partition_pattern']
seed = config['seed']
batch_size = config['batch_size']
epochs = config['epochs']
alpha = config['alpha']

taxonomy = dl.TaxoDataset(name,data_path,raw=True,partition_pattern=partition_pattern,seed=seed)
data_prep = st.Dataset(taxonomy,sampling_method,neg_number,seed)
# logging.info('data_prep.definitions')
# logging.info('data_prep.definitions')
# logging.info(data_prep.definitions)
# 11-24 15:19:51 INFO - train.py:42 - {0: {'label': 'entity||entity.n.01', 'summary': 'that which is perceived or known or inferred to have its own distinct existence (living or nonliving)'}, 1: {'label': 'physical_entity||physical_entity.n.01', 'summary': 'an entity that has physical existence'}, 2: {'label': 'abstraction||abstraction.n.06', 'summary': 'a general concept formed by extracting common features from specific examples'}, 3: {'label': 'thing||thing.n.12', 'summary': 'a separate and self-contained entity'}, 4: {'label': 'object||object.n.01', 'summary': 'a tangible and visible entity; an entity that can cast a shadow'}, 5: {'label': 'whole||whole.n.02', 'summary': 'an assemblage of parts that is regarded as a single entity'}, 6: {'label': 'congener||congener.n.03', 'summary': 'a whole (a thing or person) of the same kind or category as another'}, 7: {'label': 'living_thing||living_thing.n.01', 'summary': 'a living (or once living) entity'}, 8: {'label': 'organism||organism.n.01', 'summary': 'a living thing that has (or can develop) the ability to act or function independently'}, 9: {'label': 'benthos||benthos.n.02', 'summary': 'organisms (plants and animals) that live at or near the bottom of a sea'}, 10: {'label': 'dwarf||dwarf.n.03', 'summary': 'a plant or animal that is atypically small'}, 11: {'label': 'heterotroph||heterotroph.n.01', 'summary': 'an organism that depends on complex organic substances for nutrition'}, 12: {'label': 'parent||parent.n.02', 'summary': 'an organism (plant or animal) from which younger ones are obtained'}, 13: {'label': 'life||life.n.10', 'summary': 'living things collectively'}, 14: {'label': 'biont||biont.n.01', 'summary': 'a discrete unit of living matter'}, 15: {'label': 'cell||cell.n.02', 'summary': '(biology) the basic structural and functional unit of all organisms; they may exist as independent units of life (as in monads) or may form colonies or tissues as in higher plants and animals'}, 16: {'label': 'causal_agent||causal_agent.n.01', 'summary': 'any entity that produces an effect or is responsible for events or results'}, 17: {'label': 'person||person.n.01', 'summary': 'a human being'}, 18: {'label': 'animal||animal.n.01', 'summary': 'a living organism characterized by voluntary movement'}, 19: {'label': 'plant||plant.n.02', 'summary': '(botany) a living organism lacking the power of locomotion'}, 20: {'

model_name = config['model_name']

device = "cuda" if torch.cuda.is_available() else "cpu"
target_device = torch.device(device)

if torch.cuda.is_available():
    model = SentenceTransformer.SentenceTransformer(model_name, device='cuda')
else:
    model = SentenceTransformer.SentenceTransformer(model_name)

g = torch.Generator()
g.manual_seed(0)


nodeIdsCorpus =[data_prep.corpusId2nodeId[idx] for idx in data_prep.corpusId2nodeId]
core_graph = data_prep.core_subgraph.copy()
core_graph.remove_node(data_prep.pseudo_leaf_node)
nodes_core_subgraph = list(core_graph.nodes)
assert nodes_core_subgraph == nodeIdsCorpus
propagation = PPRPowerIteration(nx.adjacency_matrix(core_graph), alpha=alpha, niter=10).to(target_device)
logging.info('core_graph')

logging.info(type(core_graph))
logging.info(core_graph)
all_path=[]
for node, _ in core_graph.nodes(data=True):
    # if len(data_prep.definitions[node]['label']) <2: continue 
    if node == data_prep.root: continue
    if core_graph.has_node(node):
        all_path.extend(find_paths_of_length(core_graph,node, 2))
for node, _ in core_graph.nodes(data=True):
    # if len(data_prep.definitions[node]['label']) <2: continue 
    if node == data_prep.root: continue
    if core_graph.has_node(node):
        all_path.extend(find_paths_of_length(core_graph,node, 3))
with open('all_path.pkl', 'wb') as f:
    pickle.dump(all_path, f)

with open('definitions.pkl', 'wb') as f:
    pickle.dump(data_prep.definitions, f)

with open('edges.pkl', 'wb') as f:
    pickle.dump(core_graph.edges(), f)
# for edge in core_graph.edges():
#     source, target = edge
#     logging.info(f"Source: {source}, Target: {target}")
# core_graph.definitions


# for node in core_graph.nodes(data=True):
#     logging.info(node)
# Define your train dataset, the dataloader and the train loss
train_dataloader = DataLoader(data_prep.trainInput, shuffle=True, batch_size=batch_size)
warmup_steps = math.ceil(len(train_dataloader) * epochs * 0.1) #10% of train data for warm-up
train_loss = losses.CosineSimilarityLoss(model)
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(data_prep.val_examples, name='sts-dev')
# Tune the model
model.fit(train_objectives=[(train_dataloader, train_loss)], evaluator=evaluator, evaluation_steps=1000, epochs=epochs,
          warmup_steps=warmup_steps, output_path=str(config.save_dir),save_best_model=True)

model = SentenceTransformer.SentenceTransformer(str(config.save_dir))
corpus_embeddings = model.encode(data_prep.corpus, convert_to_tensor=True, show_progress_bar=True)
preds = propagation(corpus_embeddings,torch.tensor(range(len(nodeIdsCorpus)),device=target_device))

all_targets_val, all_predictions_val, all_scores_val, edges_predictions_val, all_edges_scores_val = ms.compute_prediction(data_prep.core_subgraph.edges,data_prep.pseudo_leaf_node, data_prep.valid_queries,corpus_embeddings,model,data_prep.valid_node_list,data_prep.valid_node2pos,data_prep.corpusId2nodeId)
ms.save_results(str(config.save_dir)+'/',all_targets_val, edges_predictions_val,"eval_val")
all_targets_test, all_predictions, all_scores_test, edges_predictions_test, all_edges_scores_test  = ms.compute_prediction(data_prep.core_subgraph.edges, data_prep.pseudo_leaf_node, data_prep.test_queries,corpus_embeddings,model,data_prep.test_node_list,data_prep.test_node2pos,data_prep.corpusId2nodeId)
ms.save_results(str(config.save_dir)+'/',all_targets_test, edges_predictions_test,"eval_test")


all_targets_val_ppr, all_predictions_val_ppr, all_scores_val_ppr, edges_predictions_val_ppr, all_edges_scores_val_ppr = ms.compute_prediction(data_prep.core_subgraph.edges,data_prep.pseudo_leaf_node, data_prep.valid_queries,preds,model,data_prep.valid_node_list,data_prep.valid_node2pos,data_prep.corpusId2nodeId)
ms.save_results(str(config.save_dir)+'/',all_targets_val_ppr, edges_predictions_val_ppr,"eval_val_ppr")
all_targets_test_ppr, all_predictions_ppr, all_scores_test_ppr, edges_predictions_test_ppr, all_edges_scores_test_ppr  = ms.compute_prediction(data_prep.core_subgraph.edges, data_prep.pseudo_leaf_node, data_prep.test_queries,preds,model,data_prep.test_node_list,data_prep.test_node2pos,data_prep.corpusId2nodeId)
ms.save_results(str(config.save_dir)+'/',all_targets_test_ppr, edges_predictions_test_ppr,"eval_test_ppr")