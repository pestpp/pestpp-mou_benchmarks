import pandas as pd
import numpy as np
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

temp_dir = "."

def emulate(pvals = None):
    if pvals is None:
        decvar = pd.read_csv(os.path.join(temp_dir, "dv.dat")).values.transpose()
    else:
        pvals_ordered = {pval: pvals[pval] for pval in sorted(pvals.index, key=lambda x: int(x[1:]))}
        decvar = np.array(list(pvals_ordered.values())).transpose().reshape(1, -1)
    
    training_data = pd.read_csv(os.path.join("ppd_fitness_test_obslink", "template", "trainingdata.csv"))
    y_obj_1 = training_data['obj_1'].values
    y_obj_2 = training_data['obj_2'].values
    X = training_data[['x'+str(i+1) for i in range(30)]].values

    kernel = C(1.0, (1e-1, 1e1)) * RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0))
    gpr_obj2 = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=5, random_state=42)
    gpr_obj2.fit(X, y_obj_2)

    pred_obj_2, std_2 = gpr_obj2.predict(decvar, return_std=True)

    sim = {
        'obj_1': decvar[0][0],
        'obj_2': float(pred_obj_2[0]),
        'obj_1_stdev': 0.0,
        'obj_2_stdev': float(std_2[0])
    }
    #print(sim)
    # Write output to file
    with open('gp_output.dat','w') as f:
        f.write('obsnme,obsval\n')
        f.write('obj_1,'+str(sim["obj_1"])+'\n')
        f.write('obj_2,'+str(sim["obj_2"])+'\n')
        f.write('obj_1_stdev,'+str(sim["obj_1_stdev"])+'\n')
        f.write('obj_2_stdev,'+str(sim["obj_2_stdev"])+'\n')
    
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
