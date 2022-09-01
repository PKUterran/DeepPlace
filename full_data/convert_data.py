import os
import json
import numpy as np


def node_pairs_among(nodes, max_cap=-1):
    us = []
    vs = []
    if max_cap == -1 or len(nodes) <= max_cap:
        for u in nodes:
            for v in nodes:
                if u == v:
                    continue
                us.append(u)
                vs.append(v)
    else:
        for u in nodes:
            vs_ = np.random.permutation(nodes)
            left = max_cap - 1
            for v_ in vs_:
                if left == 0:
                    break
                if u == v_:
                    continue
                us.append(u)
                vs.append(v_)
                left -= 1
    return us, vs


def convert(src_dir: str, dst_dir: str):
    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)
    for path in os.listdir(src_dir):
        data_dir = f'{src_dir}/{path}'
        dst_data_dir = f'{dst_dir}/{path}'
        if not os.path.isdir(dst_data_dir):
            os.mkdir(dst_data_dir)

        print(f'Generating {dst_data_dir}...')
        pin_net_cell = np.load(f'{data_dir}/pin_net_cell.npy')
        cell_data = np.load(f'{data_dir}/cell_data.npy')
        net_cells_dict = {}
        cell_set = set()
        for n, c in pin_net_cell:
            if cell_data[c, 3] == 0:
                continue
            net_cells_dict.setdefault(n, []).append(int(c))
            cell_set.add(int(c))

        c2c = {e: i for i, e in enumerate(cell_set)}
        net_cells_list = [[c2c[c] for c in net_cells] for net_cells in net_cells_dict.values()]
        edges_1 = []
        edges_2 = []
        for net_cells in net_cells_list:
            us, vs = node_pairs_among(net_cells, max_cap=8)
            edges_1.extend(us)
            edges_2.extend(vs)
        with open(f'{dst_data_dir}/edges_1.dat', 'w+') as fp:
            json.dump(edges_1, fp)
        with open(f'{dst_data_dir}/edges_2.dat', 'w+') as fp:
            json.dump(edges_2, fp)
        with open(f'{dst_data_dir}/n_edges.dat', 'w+') as fp:
            json.dump(net_cells_list, fp)
        with open(f'{dst_data_dir}/info.json', 'w+') as fp:
            json.dump({
                'num_cell': len(cell_set),
                'num_net': len(net_cells_list),
                'num_pin': sum([len(net_cells) for net_cells in net_cells_list]),
            }, fp)
        exit(123)


if __name__ == '__main__':
    # convert('../../Placement-datasets/dac2012', 'dac2012')
    convert('../../Placement-datasets/ispd2015', 'ispd2015')
