import os
import re
import json
import argparse
import matplotlib.pyplot as plt


def parse_data(mode='causal'):  # ['causal', 'noncausal', 'fwd+bwd']
    seqlen = [2048, 4096, 8192, 16384, 32768]
    if mode == 'fwd+bwd':
        files = [file for file in os.listdir('.') if file.endswith('.log') and 'fwd+bwd' in file]
    elif mode == 'causal':
        files = [file for file in os.listdir('.') if file.endswith('_causal.log')]
    elif mode == 'noncausal':
        files = [file for file in os.listdir('.') if file.endswith('.log') and not file.endswith('_causal.log') and not 'fwd+bwd' in file]
    
    json_data = {}
    xs, ys, ls = [], [], []
    for i, file in enumerate(files):
        raw = ''
        with open(file, 'r') as f:
            raw = f.read()
        if mode == 'causal' or mode == 'noncausal':
            label = file.split('_')[2]
            data = re.findall(r'([0-9]+.[0-9]+) ([m]*s)', raw)
            print(data, file, end='\n\n')
            assert len(data) == 14
            # data = data[::2]
            # assert len(data) == 7
            data = data[:-4:2]
            assert len(data) == 5
            data = [1000*float(data[i][0]) if data[i][1]=='s' else float(data[i][0]) for i in range(len(data))]
        else:
            raise NotImplementedError
        json_data[label] = {k: v for k, v in zip(seqlen, data)}
        xs.append(seqlen)
        ys.append(data)
        ls.append(label)
    print(xs, end='\n\n')
    print(ys, end='\n\n')
    print(ls, end='\n\n')
    json.dump(json_data, open(f'json_data_{mode}.json', 'w'), indent=4, ensure_ascii=False)
    return xs, ys, ls


def plot(xs, ys, ls, path, dpi=1000):
    fig, ax = plt.subplots()
    plt.xlabel('Sequence Length')
    plt.ylabel('Time (ms)')
    plt.title('Flash Attention Block Sparse Forward Speed')
    for x, y, l in zip(xs, ys, ls):
        ax.semilogx(x, y, label=l, linewidth=0.8, base=2)
    ax.grid(linestyle='--', linewidth=0.5)
    # ax.set(xlim=(0, 8), xticks=np.arange(1, 8), ylim=(0, 8), yticks=np.arange(1, 8))
    ax.legend(title='Sparsity Config')
    plt.show()
    plt.savefig(path, dpi=dpi)
    
    
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--mode', type=str, default='causal', choices=['causal', 'noncausal', 'fwd+bwd'])
    args = args.parse_args()
    
    xs, ys, ls = parse_data(args.mode)
    plot(xs, ys, ls, f'plot_{args.mode}_front.png', 1000)