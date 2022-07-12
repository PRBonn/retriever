import numpy as np
import matplotlib.pyplot as plt
import retriever.utils.utils as utils
import glob


if __name__ == '__main__':
    vis_all = False

    exp_dir = utils.CONFIG_DIR+'../experiments/'
    eval_files = glob.glob(exp_dir+'**/oxford_evaluation_query.txt',recursive=True)
    plt.figure()
    plt.xlabel('in top k')
    plt.ylabel('recall')
    if vis_all:
        for f in eval_files:
            label  = f.split('/')[-5:-2]
            d = np.loadtxt(f)
            plt.plot(np.arange(d.shape[0])+1,d,label=label)
    else:
        exp = {}
        for f in eval_files:
            label  = f.split('/')[-5:-2]
            label = label[0]
            d = np.loadtxt(f)

            results = {'x':np.arange(d.shape[0])+1,'y':d, 'label':f.split('/')[-5:-2],'mean':d.mean()}
            if label in exp:
                if exp[label]['mean'] < results['mean']:
                    exp[label]=results
            else:
                exp[label]= results
        for k in exp:
            plt.plot(exp[k]['x'],exp[k]['y'],label=exp[k]['label'])
    plt.legend()
    plt.grid()
    plt.show()