import torch
import torch.nn as nn
import torch.nn.functional as F


def pairwise_cosine_similarity(x, y):
    distances = torch.zeros(1).to(x.device)

    for i, x_i in enumerate(x):
        for j, y_j in enumerate(y):
            distances += (F.cosine_similarity(x_i, y_j, dim=0).to(x.device)+1e-9)
    
    #return mean of distances
    #if distances is nan print
    # if torch.isnan(distances).any():
    #     print("distances ",distances.max(),distances.min())
    return (distances / (len(x) * len(y)))+1e-9

def similar_cosine(live,attack,mix):
    # assert not torch.isnan(live).any(), "Live contains NaN values!"
    # assert not torch.isnan(attack).any(), "Attack contains NaN values!"
    # assert not torch.isnan(mix).any(), "Mix contains NaN values!"

    if torch.isnan(live).any():
        print("Live ",live.max(),live.min())
        live = torch.nan_to_num(live)
    if torch.isnan(attack).any():
        print("Attack ",attack.max(),attack.min())
        attack = torch.nan_to_num(attack)
    if torch.isnan(mix).any():
        print("Mix ",mix.max(),mix.min())
        mix = torch.nan_to_num(mix)

    # live    = F.normalize(live)
    # attack  = F.normalize(attack)
    # mix     = F.normalize(mix)

    # live    = (torch.mean(F.normalize(live, dim = 0),0))
    # attack  = (torch.mean(F.normalize(attack, dim = 0),0))
    # mix     = (torch.mean(F.normalize(mix, dim = 0),0))
    # calculate pairwise cosine similarity of live and attack
    

    l_a_cosine = pairwise_cosine_similarity(live,attack).to(live.device)
    l_m_cosine = pairwise_cosine_similarity(live,mix).to(live.device)

    # l_a_cosine = F.cosine_similarity(live,attack,dim=0).to(live.device)+1e-9
    # l_m_cosine = F.cosine_similarity(live,mix,dim=0).to(live.device)+1e-9

    # print("l_a_cosine",l_a_cosine,"l_m_cosine",l_m_cosine)

    return ((l_a_cosine-l_m_cosine).abs())