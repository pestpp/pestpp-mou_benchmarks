import pandas as pd
import numpy as np
import os
import laGPy as gpr

temp_dir = "."

def emulate(pvals = None):
    if pvals is None:
        decvar = pd.read_csv(os.path.join(temp_dir, "dv.dat")).values.transpose()
    else:
        pvals_ordered = {pval: pvals[pval] for pval in sorted(pvals.index, key=lambda x: int(x[1:]))}
        decvar = np.array(list(pvals_ordered.values())).transpose()
    
    training_data = pd.read_csv(os.path.join("ppd_fitness_test", "template", "trainingdata.csv"))
    y_obj_1 = training_data['obj_1'].values
    y_obj_2 = training_data['obj_2'].values
    X = training_data[['x'+str(i+1) for i in range(30)]].values

    pred_obj_1 = gpr.laGP(Xref=decvar, start=10, end=60, X=X, Z=y_obj_1)
    pred_obj_2 = gpr.laGP(Xref=decvar, start=10, end=60, X=X, Z=y_obj_2)
    sim = {
        'obj_1': pred_obj_1["mean"].item(),
        'obj_2': pred_obj_2["mean"].item(),
        'obj_1_sd': np.sqrt(pred_obj_1["s2"].item()),
        'obj_2_sd': np.sqrt(pred_obj_2["s2"].item())
    }

    with open('gp_output.dat','w') as f:
        f.write('obsnme,obsval\n')
        f.write('obj_1,'+str(sim["obj_1"])+'\n')
        f.write('obj_2,'+str(sim["obj_2"])+'\n')
        f.write('obj_1_sd,'+str(sim["obj_1_sd"])+'\n')
        f.write('obj_2_sd,'+str(sim["obj_2_sd"])+'\n')
    return sim

def ppw_worker(pst_name,host,port):
    import pyemu
    ppw = pyemu.os_utils.PyPestWorker(pst_name,host,port,verbose=False)
    pvals = ppw.get_parameters()
    if pvals is None:
        return

    obs = ppw._pst.observation_data.copy()
    obs = obs.loc[ppw.obs_names,"obsval"]

    while True:
        sim = emulate(pvals=pvals)
        obs.update(sim)
        ppw.send_observations(obs.values)
        pvals = ppw.get_parameters()
        if pvals is None:
            break

if __name__ == "__main__":
    emulate()
