import click
import torch
import tqdm
import retriever.datasets.datasets as datasets
import retriever.models.models as models
from torch.utils.data import DataLoader
from retriever.utils import utils
import numpy as np
import os 


def computeLatentVectors(data_loader, model):
    sequences = []
    latents = []
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader):
            latents.append(model(batch['points'].cuda()))
            sequences.append([batch['seq'].item(), batch['idx'].item()])
    sequences = torch.tensor(sequences)
    latents = torch.stack(latents).squeeze()
    return sequences, latents


@click.command()
# Add your options here
@click.option('--checkpoint',
              '-ckpt',
              type=str,
              help='path to checkpoint file (.ckpt)',
              required=True)
@click.option('--dataset',
              '-d',
              type=str,
              default='oxford',
              help='dataset',
              required=False)
@click.option('--base_dir',
              '-b',
              type=str,
              default='/data',
              help='dataset',
              required=True)
def main(checkpoint, dataset, base_dir):
    model = models.PNPerceiverModule.load_from_checkpoint(checkpoint_path=checkpoint).cuda()
    model.eval()
    
    database_file = f'{base_dir}/{dataset}_evaluation_database.pickle'
    db_dataset = datasets.OxfordQueryEmbLoaderPad(
        query_dict=database_file, data_dir=base_dir)
    database_loader = DataLoader(dataset=db_dataset, batch_size=1,
                                 shuffle=False, num_workers=0,)
    db_seq, db_latents = computeLatentVectors(database_loader, model)
    print(db_latents.shape, db_seq.shape)

    query_file = f'{base_dir}/{dataset}_evaluation_query.pickle'
    q_dataset = datasets.OxfordQueryEmbLoaderPad(
        query_dict=query_file, data_dir=base_dir)
    query_loader = DataLoader(dataset=q_dataset, batch_size=1,
                              shuffle=False, num_workers=0)
    query_seq, query_latents = computeLatentVectors(query_loader, model)

    unique_seq = query_seq[:, 0].unique().tolist()
    print(unique_seq)
    top_k = 25
    top_k_recall = torch.zeros(top_k,device=query_latents.device)
    top_one_pct_recall = 0
    counter = 0
    for seq_i in tqdm.tqdm(unique_seq):  # get all queries from seq i
        q = query_latents[query_seq[:, 0] == seq_i]
        for seq_j in (unique_seq):  # get all latents for each other seq
            if not (seq_i == seq_j):
                db = db_latents[db_seq[:, 0] == seq_j]
                one_pct_idx = max(int(np.round(db.shape[0]/100.0)), 1)-1
                # query_nn = keops.kNN_Keops(db, q, K=25, metric='euclidean')
                query_nn = utils.knn(q, db, k=25)
                # Average over each sequence
                in_top_k = torch.zeros(top_k, device=query_latents.device)
                in_one_pct = 0

                num_pos = 0
                # for each query compute metrics
                for scan_id in range(q.shape[0]):
                    true_positives = q_dataset.getTruePositives(
                        seq_i, scan_id, target_seq=seq_j)
                    true_positives = torch.tensor(
                        true_positives, dtype=query_nn.dtype, device=query_nn.device)
                    if len(true_positives) < 1:
                        continue
                    num_pos += 1
                    is_in = query_nn[scan_id:scan_id+1,
                                     :] == true_positives.unsqueeze(1)
                    is_in = is_in.any(axis=0).cumsum(0).clamp(max=1)
                    in_top_k += is_in
                    in_one_pct += is_in[one_pct_idx]
                top_k_recall += in_top_k / num_pos
                top_one_pct_recall += in_one_pct/num_pos
                counter += 1

    top_k_recall/=counter
    top_one_pct_recall/=counter
    print(top_k_recall)
    print(top_one_pct_recall)
    out_f = os.path.dirname(os.path.abspath(checkpoint))
    out_name = os.path.join(out_f,query_file.split('/')[-1].split('.')[0]+'.txt')
    np.savetxt(out_name,top_k_recall.cpu().numpy(),fmt='%.9f',
                header=f'#Top 1 percent recall:\n{top_one_pct_recall}\n#Top k:')

if __name__ == "__main__":
    with torch.no_grad():
        main()
