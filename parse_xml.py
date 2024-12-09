# https://blog.kobalab.net/entry/20170225/1488036549

import xml.etree.ElementTree as ET
from collections import Counter
import os
import torch

def dict_to_features(pl, target, tehais, state, current_round):
    #Sparse/tile features
    opps = [x for x in range(4) if x != pl]
    feature = torch.zeros(34,146, dtype=torch.int16)
    feature[target // 4, 0] = 1
    count = Counter([tile//4 for tile in tehais[pl]])
    for tile, c in count.items():
        feature[tile, 1] = c
    if 16 in count:
        feature[4, 2] = 1
    if 52 in count:
        feature[13, 2] = 1
    if 88 in count:
        feature[22, 2] = 1
    
    #In the paper, this takes 36 channels... but not really sure how that works?
    #My version takes 1 column for each furo
    for idx, furo in enumerate(state["furos"][pl]):
        furo_type = furo.split()[0]
        furo_tiles = furo.split()[1].split(",")
        if furo_type == "chi":
            for tile in furo_tiles:
                feature[int(tile)//4, 3+idx] = 1
        elif furo_type == "pong":
            feature[int(furo_tiles[0])//4, 3+idx] = 3
        else:
            feature[int(furo_tiles[0])//4, 3+idx] = 4
    
    for idx, discard in enumerate(state["discards"][pl]):
        feature[discard//4, 7+idx] #columns 7 ~36
    for idx, dora_indicator in enumerate(state["dora_indicators"]):
        feature[dora_indicator//4, 37+idx] += 1 #columns 37 ~ 41
    
    #Other players' open tiles
    base = 42
    for opp in opps:
        for idx, furo in enumerate(state["furos"][opp]):
            furo_type = furo.split()[0]
            furo_tiles = furo.split()[1].split(",")
            if furo_type == "chi":
                for tile in furo_tiles:
                    feature[int(tile)//4, base+idx] = 1
            elif furo_type == "pong":
                feature[int(furo_tiles[0])//4, base+idx] = 3
            else:
                feature[int(furo_tiles[0])//4, base+idx] = 4
        base += 4
        
        for idx, discard in enumerate(state["discards"][opp]):
            feature[discard//4, base+idx]
        base += 30
    feature = feature.to_sparse()
    
    #Dense features
    feature_dense = torch.zeros(12, dtype=torch.int16)
    for i in range(3):
        feature_dense[i] = state["riichi"][opps[i]]
    for i in range(3, 7):
        feature_dense[i] = state["scores"][i-3]
    feature_dense[7] = current_round["kyoku"]
    feature_dense[8] = current_round["honba"]
    feature_dense[9] = state["kyotaku"]
    feature_dense[10] = pl == current_round["oya"]
    feature_dense[11] = pl

    return feature, feature_dense

def parse_xml(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    #root[0]: SHUFFLE. ignore.
    #root[1]: GO. game settings.
    n = int(root[1].get('type'))
    if n & 0x10 or n & 0x02:
        return None #Ignore games that are sanma/have no akadora
    #root[2]: UN (user info) ignore.
    #root[3]: TAIKYOKU. oya = 0. ignore.
    #From here, repeat INIT - TUVW & DEFG, N, DORA, REACH, AGARI/RYUUKYOKU
    discard_data = [[], [], []] # Sparse features, dense features, label
    riichi_data = [[], [], []]
    chi_data = [[], [], []]
    pong_data = [[], [], []]
    kang_data = [[], [], []]
    riichied = False
    # rewards = [[], [], []] #sparse features, dense features, reward

    for child in root[4:]:
        if child.tag == "INIT":
            #Fixed for this round
            current_round = {}
            seed = child.get("seed").split(",") #seed[3], seed[4], the dice results, aren't important
            current_round["kyoku"] = int(seed[0]) #East 1 = 0, East 2 = 1, ...
            current_round["honba"] = int(seed[1])
            current_round["oya"] = int(child.get("oya"))

            #Not fixed, but public info
            state = {"scores": [int(x) for x in child.get("ten").split(",")],
                     "kyotaku": int(seed[2]),
                     "dora_indicators": [int(seed[5])],
                     "discards": [[] for _ in range(4)],
                     "furos": [[] for _ in range(4)],
                     "riichi": [False for _ in range(4)]}
            
            #Not fixed, not public
            tehais = [[int(x) for x in child.get("hai0").split(",")],
                     [int(x) for x in child.get("hai1").split(",")],
                     [int(x) for x in child.get("hai2").split(",")],
                     [int(x) for x in child.get("hai3").split(",")]]

        elif child.tag == "DORA":
            state["dora_indicators"].append(int(child.get("hai")))
        elif child.tag == "BYE" or child.tag == "UN": # I think this has to do with a connection problem? but unsure
            pass
        elif child.tag[0] in "TUVW": #Tsumo
            pl = ord(child.tag[0])-ord("T")
            tsumo = int(child.tag[1:])
            tehais[pl].append(tsumo)
        elif child.tag[0] in "DEFG": #Discard
            pl = ord(child.tag[0])-ord("D")
            discard = int(child.tag[1:])
            discard_vec = torch.zeros(34, dtype=torch.int16)
            discard_vec[discard//4] = 1
            sparse, dense = dict_to_features(pl, tsumo, tehais, state, current_round)
            discard_data[0].append(sparse)
            discard_data[1].append(dense)
            discard_data[2].append(discard_vec)
            if not riichied:
                riichi_data[0].append(sparse)
                riichi_data[1].append(dense)
                riichi_data[2].append(torch.tensor(0))
            state["discards"][pl].append(discard)
            tehais[pl].remove(discard)
        elif child.tag == "N":
            pl = int(child.get("who"))
            menzi = int(child.get("m"))
            if menzi & 0x0004: #shunzi
                p = ((menzi & 0xFC00) >> 10) // 3
                suit = p // 7
                num = p % 7
                start = suit*9 + num
                sparse, dense = dict_to_features(pl, discard, tehais, state, current_round)
                label = torch.zeros(34, dtype=torch.int16)
                label[start] = 1 #Starting tile of chi
                chi_data[0].append(sparse)
                chi_data[1].append(dense)
                chi_data[2].append(label)

                #negatives
                for opp in [x for x in range(4) if x != pl]:
                    sparse, dense = dict_to_features(opp, discard, tehais, state, current_round)
                    label = torch.zeros(34, dtype=torch.int16)
                    chi_data[0].append(sparse)
                    chi_data[1].append(dense)
                    chi_data[2].append(label)
                    pong_data[0].append(sparse)
                    pong_data[1].append(dense)
                    pong_data[2].append(torch.tensor(0))
                    kang_data[0].append(sparse)
                    kang_data[1].append(dense)
                    kang_data[2].append(torch.tensor(0))

                chi = "chi "
                for additive in [(menzi & 0x18)>>3, (menzi & 0x60)>>5, (menzi & 0x180)>>7]:
                    if start*4 + additive != discard:
                        tehais[pl].remove(start*4+additive)
                    chi += str(start+additive)+","
                    start += 1
                chi = chi[:-1]
                state["furos"][pl].append(chi)
                
            elif menzi & 0x0018: #kezi, Addkang
                if menzi & 0x0010: #Addkang
                    sparse, dense = dict_to_features(pl, tsumo, tehais, state, current_round)
                    kang_data[0].append(sparse)
                    kang_data[1].append(dense)
                    kang_data[2].append(torch.tensor(1))
                    start = ((menzi & 0xFE00) >> 9) // 3 * 4
                    result = [i for i in range(len(state["furos"][pl])) if state["furos"][pl][i].startswith("pong") and \
                              (str(start) in state["furos"][pl][i].split()[1].split(",") or \
                               str(start+1) in state["furos"][pl][i].split()[1].split(","))]
                    state["furos"][pl][result[0]] = "kang "+",".join([str(start+i) for i in range(4)])
                    
                else:
                    sparse, dense = dict_to_features(pl, discard, tehais, state, current_round)
                    pong_data[0].append(sparse)
                    pong_data[1].append(dense)
                    pong_data[2].append(torch.tensor(1))
                    
                    #negatives
                    for opp in [x for x in range(4) if x != pl]:
                        sparse, dense = dict_to_features(opp, discard, tehais, state, current_round)
                        label = torch.zeros(34, dtype=torch.int16)
                        chi_data[0].append(sparse)
                        chi_data[1].append(dense)
                        chi_data[2].append(label)
                        pong_data[0].append(sparse)
                        pong_data[1].append(dense)
                        pong_data[2].append(torch.tensor(0))
                        kang_data[0].append(sparse)
                        kang_data[1].append(dense)
                        kang_data[2].append(torch.tensor(0))
                
                    pong = "pong "
                    for i in range(4):
                        tile = discard - (discard % 4) + i
                        if i != (menzi & 0x0060) >> 5 and tile != discard:
                            tehais[pl].remove(tile)
                            pong += str(tile)+","
                    pong += str(discard)
                    state["furos"][pl].append(pong)
            else: #Closed/Open kang
                if menzi & 0x0003: #Open kang
                    target = discard
                else: #Closed kang
                    target = (menzi & 0xFF00) >> 8
                sparse, dense = dict_to_features(pl, target, tehais, state, current_round)
                kang_data[0].append(sparse)
                kang_data[1].append(dense)
                kang_data[2].append(torch.tensor(1))

                kang = "kang "
                for i in range(4):
                    tile = target - (target % 4) + i
                    if tile in tehais[pl]:
                        tehais[pl].remove(tile)
                    kang += str(tile)+","
                kang = kang[:-1]
                state["furos"].append(kang)
            
        elif child.tag == "REACH":
            pl = int(child.get("who"))
            if child.get("step") == "1":
                sparse, dense = dict_to_features(pl, tsumo, tehais, state, current_round)
                riichi_data[0].append(sparse)
                riichi_data[1].append(dense)
                riichi_data[2].append(torch.tensor(1))
                riichied = True
            else:
                state["scores"] = [int(x) for x in child.get("ten").split(",")]
                riichied = False
                state["riichi"][pl] = True
                state["kyotaku"] += 1
        elif child.tag == "AGARI" or child.tag == "RYUUKYOKU":
            pass
            # rewards = [int(x)*100 for x in child.get("sc").split(",")[1::2]]
        else:
            print(child.tag, child.attrib, filename)
            raise NotImplementedError
    
    if discard_data[0]:
        for idx in range(3):
            discard_data[idx] = torch.stack(discard_data[idx])
    else:
        discard_data[0] = torch.zeros(0,34,146, dtype=torch.int16).to_sparse()
        discard_data[1] = torch.zeros(0,12, dtype=torch.int16)
        discard_data[2] = torch.zeros(0,34, dtype=torch.int16)

    if riichi_data[0]:
        for idx in range(3):
            riichi_data[idx] = torch.stack(riichi_data[idx])
    else:
        riichi_data[0] = torch.zeros(0,34,146, dtype=torch.int16).to_sparse()
        riichi_data[1] = torch.zeros(0,12, dtype=torch.int16)
        riichi_data[2] = torch.zeros(0, dtype=torch.int16)

    if chi_data[0]:
        for idx in range(3):
            chi_data[idx] = torch.stack(chi_data[idx])
    else:
        chi_data[0] = torch.zeros(0,34,146, dtype=torch.int16).to_sparse()
        chi_data[1] = torch.zeros(0,12, dtype=torch.int16)
        chi_data[2] = torch.zeros(0,34, dtype=torch.int16)
        
    if pong_data[0]:
        for idx in range(3):
            pong_data[idx] = torch.stack(pong_data[idx])
    else:
        pong_data[0] = torch.zeros(0,34,146, dtype=torch.int16).to_sparse()
        pong_data[1] = torch.zeros(0,12, dtype=torch.int16)
        pong_data[2] = torch.zeros(0, dtype=torch.int16)

    if kang_data[0]:
        for idx in range(3):
            kang_data[idx] = torch.stack(kang_data[idx])
    else:
        kang_data[0] = torch.zeros(0,34,146, dtype=torch.int16).to_sparse()
        kang_data[1] = torch.zeros(0,12, dtype=torch.int16)
        kang_data[2] = torch.zeros(0, dtype=torch.int16)
    
    return discard_data, riichi_data, chi_data, pong_data, kang_data


folderpath = 'data/'
out_dir = 'features/'

discard_data = [[], [], []]
riichi_data = [[], [], []]
chi_data = [[], [], []]
pong_data = [[], [], []]
kang_data = [[], [], []]

lengths = [[] for _ in range(5)]

day = "20230101"

for file in os.listdir(folderpath):
    if day not in file:
        for idx in range(3):
            discard_data[idx] = torch.cat(discard_data[idx])
            riichi_data[idx] = torch.cat(riichi_data[idx])
            chi_data[idx] = torch.cat(chi_data[idx])
            pong_data[idx] = torch.cat(pong_data[idx])
            kang_data[idx] = torch.cat(kang_data[idx])

        torch.save(discard_data, out_dir+"discard"+day+".pt")
        lengths[0].append(discard_data[0].shape[0])
        torch.save(riichi_data, out_dir+"riichi"+day+".pt")
        lengths[1].append(riichi_data[0].shape[0])
        torch.save(chi_data, out_dir+"chi"+day+".pt")
        lengths[2].append(chi_data[0].shape[0])
        torch.save(pong_data, out_dir+"pong"+day+".pt")
        lengths[3].append(pong_data[0].shape[0])
        torch.save(kang_data, out_dir+"kang"+day+".pt")
        lengths[4].append(kang_data[0].shape[0])

        day = file[:8]

        discard_data = [[], [], []]
        riichi_data = [[], [], []]
        chi_data = [[], [], []]
        pong_data = [[], [], []]
        kang_data = [[], [], []]

    data = parse_xml(folderpath+file)
    if data is None:
        continue
    discard, riichi, chi, pong, kang = data

    for idx in range(3):
        discard_data[idx].append(discard[idx])
        riichi_data[idx].append(riichi[idx])
        chi_data[idx].append(chi[idx])
        pong_data[idx].append(pong[idx])
        kang_data[idx].append(kang[idx])

for idx in range(3):
    discard_data[idx] = torch.cat(discard_data[idx])
    riichi_data[idx] = torch.cat(riichi_data[idx])
    chi_data[idx] = torch.cat(chi_data[idx])
    pong_data[idx] = torch.cat(pong_data[idx])
    kang_data[idx] = torch.cat(kang_data[idx])


torch.save(discard_data, out_dir+"discard"+day+".pt")
lengths[0].append(discard_data[0].shape[0])
torch.save(riichi_data, out_dir+"riichi"+day+".pt")
lengths[1].append(riichi_data[0].shape[0])
torch.save(chi_data, out_dir+"chi"+day+".pt")
lengths[2].append(chi_data[0].shape[0])
torch.save(pong_data, out_dir+"pong"+day+".pt")
lengths[3].append(pong_data[0].shape[0])
torch.save(kang_data, out_dir+"kang"+day+".pt")
lengths[4].append(kang_data[0].shape[0])

for i in range(5):
    lengths[i] = torch.tensor(lengths[i])

torch.save(lengths, out_dir+"lengths.pt")
