import os
import json
import numpy as np


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
        net_cells_dict = {}
        cell_set = set()
        for n, c in pin_net_cell:
            net_cells_dict.setdefault(n, []).append(int(c))
            cell_set.add(int(c))
        ret_list = [v for v in net_cells_dict.values()]
        with open(f'{dst_data_dir}/n_edges.dat', 'w+') as fp:
            json.dump(ret_list, fp)
        with open(f'{dst_data_dir}/info.json', 'w+') as fp:
            json.dump({'num_cell': max(cell_set) + 1}, fp)


if __name__ == '__main__':
    # convert('../../Placement-datasets/dac2012', 'dac2012')
    convert('../../Placement-datasets/ispd2015', 'ispd2015')
