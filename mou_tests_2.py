import os
import sys
import shutil
import platform
import numpy as np
import pandas as pd
import platform
import pyemu

import opt_test_suite_helper as mou_suite_helper

bin_path = os.path.join("test_bin")
if "linux" in platform.platform().lower():
    bin_path = os.path.join(bin_path,"linux")
elif "darwin" in platform.platform().lower():
    bin_path = os.path.join(bin_path,"mac")
else:
    bin_path = os.path.join(bin_path,"win")

bin_path = os.path.abspath("test_bin")
os.environ["PATH"] += os.pathsep + bin_path


bin_path = os.path.join("..","..","..","bin")
exe = ""
if "windows" in platform.platform().lower():
    exe = ".exe"
exe_path = os.path.join(bin_path, "pestpp-mou" + exe)


noptmax = 4
num_reals = 20
port = 4021
test_root = "mou_tests"



def test_sorting_fake_problem():
    test_d = os.path.join(test_root,"sorting_test")
    if os.path.exists(test_d):
        shutil.rmtree(test_d)
    os.makedirs(test_d)

    with open(os.path.join(test_d,"par.tpl"),'w') as f:
        f.write("ptf ~\n ~   par1    ~\n")
        f.write("~   par2    ~\n")
        
    with open(os.path.join(test_d,"obs.ins"),'w') as f:
        f.write("pif ~\n")
        for i in range(6): # the number of objs in the test
            f.write("l1 !obj{0}!\n".format(i))
    pst = pyemu.Pst.from_io_files(os.path.join(test_d,"par.tpl"),"par.dat",os.path.join(test_d,"obs.ins"),"obs.dat",pst_path=".")
    obs = pst.observation_data
    obs.loc[:,"obgnme"] = "less_than_obj"
    obs.loc[:,"obsval"] = 0.0
    obs.loc[:,"weight"] = 1.0
    
    par = pst.parameter_data
    par.loc[:,"partrans"] = "none"
    par.loc[:,"parval1"] = 1.0
    par.loc[:,"parubnd"] = 1.5
    par.loc[:,"parlbnd"] = 0.5

    pst.control_data.noptmax = -1
    np.random.seed(111)

    pe = pyemu.ParameterEnsemble.from_gaussian_draw(pst=pst,num_reals=50)
    oe = pyemu.ObservationEnsemble.from_gaussian_draw(pst=pst,num_reals=50)

    pe.to_csv(os.path.join(test_d,"par.csv"))
    oe.to_csv(os.path.join(test_d,"obs.csv"))
    pst.pestpp_options["mou_dv_population_file"] = "par.csv"
    pst.pestpp_options["mou_obs_population_restart_file"] = "obs.csv"
    pst.write(os.path.join(test_d,"test.pst"))
    pyemu.os_utils.run("{0} test.pst".format(exe_path),cwd=test_d)

    cov = pyemu.Cov.from_parameter_data(pst).to_2d()
    cov.x[0,1] = 0.0001
    cov.x[1,0] = 0.0001


    pyemu.helpers.first_order_pearson_tikhonov(pst=pst,cov=cov,abs_drop_tol=0.0)
    print(pst.prior_information)
    #pst.prior_information = pst.prior_information.loc[["pcc_3"],:]
    pi = pst.prior_information
    pi.loc["pcc_1","equation"] = pi.loc["pcc_1","equation"].replace("= 0.0","= 1.0").replace(" - "," + ")
    pi.loc[:,"obgnme"] = "less_than_pi"
    pst.write(os.path.join(test_d,"test.pst"))
    pyemu.os_utils.run("{0} test.pst".format(exe_path),cwd=test_d)



def test_risk_obj():
    t_d = mou_suite_helper.setup_problem("zdt1",True,True)
    df = pd.read_csv(os.path.join(t_d,"prior.csv"),index_col=0)
    df.loc[:,"_risk_"] = 0.95
    print(df.columns)
    df.to_csv(os.path.join(t_d,"prior.csv"))
    pst = pyemu.Pst(os.path.join(t_d,"zdt1.pst"))
    pst.pestpp_options["mou_dv_population_file"] = "prior.csv"
    pst.pestpp_options["opt_chance_points"] = "single"
    pst.pestpp_options["opt_recalc_chance_every"] = 1000
    pst.pestpp_options["opt_stack_size"] = 10
    pst.pestpp_options["opt_par_stack"] = "prior.csv"
    pst.pestpp_options["mou_generator"] = "de"
    pst.control_data.noptmax = -1
    pst.write(os.path.join(t_d,"zdt1.pst"))
    m1 = os.path.join("mou_tests","zdt1_test_master_riskobj")
    pyemu.os_utils.start_workers(t_d,exe_path,"zdt1.pst",35,worker_root="mou_tests",
                                 master_dir=m1,verbose=True,port=port)

    t_d = mou_suite_helper.setup_problem("zdt1", True, False)
    df.pop("_risk_")
    df.to_csv(os.path.join(t_d, "prior.csv"))
    pst = pyemu.Pst(os.path.join(t_d, "zdt1.pst"))
    pst.pestpp_options["mou_dv_population_file"] = "prior.csv"
    pst.pestpp_options["opt_chance_points"] = "single"
    pst.pestpp_options["opt_recalc_chance_every"] = 1000
    pst.pestpp_options["opt_risk"] = 0.95
    pst.pestpp_options["opt_stack_size"] = 10
    pst.pestpp_options["opt_par_stack"] = "prior.csv"
    pst.pestpp_options["mou_generator"] = "de"
    pst.control_data.noptmax = -1
    pst.write(os.path.join(t_d, "zdt1.pst"))
    m2 = os.path.join("mou_tests","zdt1_test_master")
    pyemu.os_utils.start_workers(t_d, exe_path, "zdt1.pst", 35,
                                 worker_root="mou_tests",
                                 master_dir=m2,
                                 verbose=True,
                                 port=port)

    test_files = ["zdt1.0.obs_pop.csv","zdt1.0.obs_stack.csv","zdt1.0.obs_pop.chance.csv"]
    for test_file in test_files:
        df1 = pd.read_csv(os.path.join(m1,test_file),index_col=0)
        df2 = pd.read_csv(os.path.join(m2, test_file), index_col=0)
        d = (df1 - df2).apply(np.abs)
        print(d.max().max())


def test_restart_single():
    t_d = mou_suite_helper.setup_problem("zdt1", True, True)
    pst = pyemu.Pst(os.path.join(t_d, "zdt1.pst"))
    pst.pestpp_options["opt_chance_points"] = "single"
    pst.pestpp_options["opt_recalc_chance_every"] = 1000
    pst.pestpp_options["opt_stack_size"] = 10
    pst.pestpp_options["mou_population_size"] = 10
    pst.pestpp_options["opt_par_stack"] = "prior.csv"
    pst.pestpp_options["mou_generator"] = "de"
    pst.control_data.noptmax = -1
    pst.write(os.path.join(t_d, "zdt1.pst"))
    m1 = os.path.join("mou_tests", "zdt1_test_master_restart1")
    pyemu.os_utils.start_workers(t_d, exe_path, "zdt1.pst", 35, worker_root="mou_tests",
                                 master_dir=m1, verbose=True,port=port)

    shutil.copy2(os.path.join(m1,'zdt1.0.par_stack.csv'),os.path.join(t_d,"par_stack.csv"))
    shutil.copy2(os.path.join(m1, 'zdt1.0.obs_stack.csv'), os.path.join(t_d, "obs_stack.csv"))


    pst.pestpp_options["opt_par_stack"] = "par_stack.csv"
    pst.pestpp_options["opt_obs_stack"] = "obs_stack.csv"
    pst.control_data.noptmax = 3
    pst.pestpp_options["opt_recalc_chance_every"] = 2
    pst.write(os.path.join(t_d, "zdt1.pst"))
    m2 = os.path.join("mou_tests", "zdt1_test_master_restart2")
    pyemu.os_utils.start_workers(t_d, exe_path, "zdt1.pst", 35, worker_root="mou_tests",
                                 master_dir=m2, verbose=True,port=port)

    chance_file = "zdt1.0.obs_pop.chance.csv"
    d1 = pd.read_csv(os.path.join(m1,chance_file),index_col=0)
    d2 = pd.read_csv(os.path.join(m2, chance_file), index_col=0)
    d = (d1-d2).apply(np.abs)
    print(d.max().max())
    assert d.max().max() < 0.01

def test_restart_all():
    t_d = mou_suite_helper.setup_problem("zdt1", True, True)
    pst = pyemu.Pst(os.path.join(t_d, "zdt1.pst"))
    pst.pestpp_options["opt_chance_points"] = "all"
    pst.pestpp_options["opt_recalc_chance_every"] = 1000
    pst.pestpp_options["opt_stack_size"] = 5
    pst.pestpp_options["mou_population_size"] = 10
    pst.pestpp_options["opt_par_stack"] = "prior.csv"
    pst.pestpp_options["mou_generator"] = "de"
    pst.control_data.noptmax = -1
    pst.write(os.path.join(t_d, "zdt1.pst"))
    m1 = os.path.join("mou_tests", "zdt1_test_master_restart1")
    pyemu.os_utils.start_workers(t_d, exe_path, "zdt1.pst", 35, worker_root="mou_tests",
                                 master_dir=m1, verbose=True,port=port)

    shutil.copy2(os.path.join(m1, 'zdt1.0.nested.par_stack.csv'), os.path.join(t_d, "par_stack.csv"))
    shutil.copy2(os.path.join(m1, 'zdt1.0.nested.obs_stack.csv'), os.path.join(t_d, "obs_stack.csv"))

    pst.pestpp_options["opt_par_stack"] = "par_stack.csv"
    pst.pestpp_options["opt_obs_stack"] = "obs_stack.csv"
    pst.control_data.noptmax = 3
    pst.pestpp_options["opt_recalc_chance_every"] = 2
    pst.write(os.path.join(t_d, "zdt1.pst"))
    m2 = os.path.join("mou_tests", "zdt1_test_master_restart2")
    pyemu.os_utils.start_workers(t_d, exe_path, "zdt1.pst", 35, worker_root="mou_tests",
                                 master_dir=m2, verbose=True,port=port)

    chance_file = "zdt1.0.obs_pop.chance.csv"
    d1 = pd.read_csv(os.path.join(m1, chance_file), index_col=0)
    d2 = pd.read_csv(os.path.join(m2, chance_file), index_col=0)
    d = (d1 - d2).apply(np.abs)
    print(d.max().max())
    assert d.max().max() < 0.01

    t_d = mou_suite_helper.setup_problem("zdt1", True, True)
    pst = pyemu.Pst(os.path.join(t_d, "zdt1.pst"))
    pst.pestpp_options["opt_chance_points"] = "all"
    pst.pestpp_options["opt_recalc_chance_every"] = 1000
    pst.pestpp_options["opt_stack_size"] = 5
    pst.pestpp_options["mou_population_size"] = 10
    pst.pestpp_options["opt_par_stack"] = "prior.csv"
    pst.pestpp_options["mou_generator"] = "pso"
    pst.control_data.noptmax = -1
    pst.write(os.path.join(t_d, "zdt1.pst"))
    m1 = os.path.join("mou_tests", "zdt1_test_master_restart1")
    pyemu.os_utils.start_workers(t_d, exe_path, "zdt1.pst", 35, worker_root="mou_tests",
                                 master_dir=m1, verbose=True,port=port)

    shutil.copy2(os.path.join(m1, 'zdt1.0.nested.par_stack.csv'), os.path.join(t_d, "par_stack.csv"))
    shutil.copy2(os.path.join(m1, 'zdt1.0.nested.obs_stack.csv'), os.path.join(t_d, "obs_stack.csv"))

    pst.pestpp_options["opt_par_stack"] = "par_stack.csv"
    pst.pestpp_options["opt_obs_stack"] = "obs_stack.csv"
    pst.control_data.noptmax = 3
    pst.pestpp_options["opt_recalc_chance_every"] = 2
    pst.write(os.path.join(t_d, "zdt1.pst"))
    m2 = os.path.join("mou_tests", "zdt1_test_master_restart2")
    pyemu.os_utils.start_workers(t_d, exe_path, "zdt1.pst", 35, worker_root="mou_tests",
                                 master_dir=m2, verbose=True,port=port)

    chance_file = "zdt1.0.obs_pop.chance.csv"
    d1 = pd.read_csv(os.path.join(m1, chance_file), index_col=0)
    d2 = pd.read_csv(os.path.join(m2, chance_file), index_col=0)
    d = (d1 - d2).apply(np.abs)
    print(d.max().max())
    assert d.max().max() < 0.01

def invest_risk_obj():
    t_d = mou_suite_helper.setup_problem("zdt1",True,True)
    pst = pyemu.Pst(os.path.join(t_d,"zdt1.pst"))
    pst.pestpp_options["opt_chance_points"] = "single"
    pst.pestpp_options["opt_recalc_chance_every"] = 1000
    pst.pestpp_options["opt_stack_size"] = 50
    pst.pestpp_options["mou_generator"] = "de"
    pst.control_data.noptmax = 300
    pst.write(os.path.join(t_d,"zdt1.pst"))
    m1 = os.path.join("mou_tests","zdt1_test_master_riskobj_full")
    pyemu.os_utils.start_workers(t_d,exe_path,"zdt1.pst",35,worker_root="mou_tests",
                                 master_dir=m1,verbose=True,port=port)
    plot_results(os.path.join("mou_tests", "zdt1_test_master_riskobj_full"))

def chance_consistency_test():
    t_d = mou_suite_helper.setup_problem("constr", additive_chance=True, risk_obj=False)
    pst = pyemu.Pst(os.path.join(t_d, "constr.pst"))
    pst.pestpp_options["opt_chance_points"] = "all"
    pst.pestpp_options["opt_recalc_chance_every"] = 5
    pst.pestpp_options["opt_stack_size"] = 10
    pst.pestpp_options["mou_generator"] = "de"
    pst.pestpp_options["mou_population_size"] = 10
    pst.pestpp_options["opt_risk"] = 0.95
    pst.control_data.noptmax = -1
    pst.write(os.path.join(t_d, "constr.pst"))
    m1 = os.path.join("mou_tests", "constr_test_master_fail_3")
    pyemu.os_utils.start_workers(t_d, exe_path, "constr.pst", 35, worker_root="mou_tests",
                                 master_dir=m1, verbose=True,port=port)


    
    tag = "nearest-to-mean single point"
    with open(os.path.join(m1,"constr.rec"),'r') as f:
        while True:
            line = f.readline()
            if line == "":
                raise Exception()
            if tag in line:
                mname = line.strip().split()[2].lower()
                for _ in range(4):
                    f.readline()
                o1 = float(f.readline().strip().split()[-1])
                o2 = float(f.readline().strip().split()[-1])
                break
        

    print(mname,o1,o2)
    df = pd.read_csv(os.path.join(m1,"constr.0.obs_pop.chance.csv"),index_col=0)
    d1 = np.abs(df.loc[mname,:].iloc[0] - o1)
    d2 = np.abs(df.loc[mname,:].iloc[1] - o2)
    print(mname,o1,d1,o2,d2)
    assert d1 < 1.e-6
    assert d2 < 1.e-6
    
def fail_test():
    t_d = mou_suite_helper.setup_problem("zdt1", additive_chance=True, risk_obj=False)
    pst = pyemu.Pst(os.path.join(t_d, "zdt1.pst"))
    pst.pestpp_options["opt_chance_points"] = "all"
    pst.pestpp_options["opt_recalc_chance_every"] = 5
    pst.pestpp_options["opt_stack_size"] = 10
    pst.pestpp_options["mou_generator"] = "de"
    pst.pestpp_options["mou_population_size"] = 30
    pst.pestpp_options["ies_debug_fail_remainder"] = True
    pst.pestpp_options["opt_risk"] = 0.95
    pst.control_data.noptmax = 8
    pst.write(os.path.join(t_d, "zdt1.pst"))
    m1 = os.path.join("mou_tests", "zdt1_test_master_fail_1")
    pyemu.os_utils.start_workers(t_d, exe_path, "zdt1.pst", 35, worker_root="mou_tests",
                                 master_dir=m1, verbose=True,port=port)

    dp_file = os.path.join(m1,"zdt1.0.nested.par_stack.csv")
    dp = pd.read_csv(dp_file,index_col=0)
    op_file = os.path.join(m1,"zdt1.0.nested.obs_stack.csv")
    op = pd.read_csv(dp_file,index_col=0)
    print(dp.shape,op.shape)
    assert dp.shape[0] == op.shape[0]
    
    shutil.copy2(dp_file,os.path.join(t_d,"par_stack.csv"))
    shutil.copy2(op_file,os.path.join(t_d,"obs_stack.csv"))
    pst.pestpp_options["opt_par_stack"] = "par_stack.csv"
    pst.pestpp_options["opt_obs_stack"] = "obs_stack.csv"
    pst.control_data.noptmax = 2
    pst.write(os.path.join(t_d, "zdt1.pst"))
    m1 = os.path.join("mou_tests", "zdt1_test_master_fail_1")
    pyemu.os_utils.start_workers(t_d, exe_path, "zdt1.pst", 35, worker_root="mou_tests",
                                 master_dir=m1, verbose=True,port=port)


    t_d = mou_suite_helper.setup_problem("zdt1", additive_chance=True, risk_obj=False)
    pst = pyemu.Pst(os.path.join(t_d, "zdt1.pst"))
    pst.pestpp_options["opt_chance_points"] = "all"
    pst.pestpp_options["opt_recalc_chance_every"] = 5
    pst.pestpp_options["opt_stack_size"] = 10
    pst.pestpp_options["mou_generator"] = "de"
    pst.pestpp_options["mou_population_size"] = 30
    pst.pestpp_options["ies_debug_fail_subset"] = True
    pst.pestpp_options["opt_risk"] = 0.95
    pst.control_data.noptmax = 8
    pst.write(os.path.join(t_d, "zdt1.pst"))
    m1 = os.path.join("mou_tests", "zdt1_test_master_fail_1")
    pyemu.os_utils.start_workers(t_d, exe_path, "zdt1.pst", 35, worker_root="mou_tests",
                                 master_dir=m1, verbose=True)


    dp_file = os.path.join(m1,"zdt1.0.nested.par_stack.csv")
    dp = pd.read_csv(dp_file,index_col=0)
    op_file = os.path.join(m1,"zdt1.0.nested.obs_stack.csv")
    op = pd.read_csv(dp_file,index_col=0)
    print(dp.shape,op.shape)
    assert dp.shape[0] == op.shape[0]
    
    shutil.copy2(dp_file,os.path.join(t_d,"par_stack.csv"))
    shutil.copy2(op_file,os.path.join(t_d,"obs_stack.csv"))
    pst.pestpp_options["opt_par_stack"] = "par_stack.csv"
    pst.pestpp_options["opt_obs_stack"] = "obs_stack.csv"
    pst.control_data.noptmax = 2
    pst.write(os.path.join(t_d, "zdt1.pst"))
    m1 = os.path.join("mou_tests", "zdt1_test_master_fail_1")
    pyemu.os_utils.start_workers(t_d, exe_path, "zdt1.pst", 35, worker_root="mou_tests",
                                 master_dir=m1, verbose=True,port=port)

    t_d = mou_suite_helper.setup_problem("zdt1", additive_chance=True,risk_obj=False)
    pst = pyemu.Pst(os.path.join(t_d, "zdt1.pst"))
    pst.pestpp_options["opt_chance_points"] = "single"
    pst.pestpp_options["opt_recalc_chance_every"] = 1000
    pst.pestpp_options["opt_stack_size"] = 10
    pst.pestpp_options["mou_generator"] = "de"
    pst.pestpp_options["mou_population_size"] = 30
    pst.pestpp_options["ies_debug_fail_subset"] = True
    pst.pestpp_options["opt_risk"] = 0.95
    pst.control_data.noptmax = 8
    pst.write(os.path.join(t_d, "zdt1.pst"))
    m1 = os.path.join("mou_tests", "zdt1_test_master_fail_1")
    pyemu.os_utils.start_workers(t_d, exe_path, "zdt1.pst", 35, worker_root="mou_tests",
                                 master_dir=m1, verbose=True,port=port)


def pso_fail_test():
    t_d = mou_suite_helper.setup_problem("zdt1", additive_chance=True, risk_obj=False)
    pst = pyemu.Pst(os.path.join(t_d, "zdt1.pst"))
    pst.pestpp_options["opt_chance_points"] = "all"
    pst.pestpp_options["opt_recalc_chance_every"] = 5
    pst.pestpp_options["opt_stack_size"] = 10
    pst.pestpp_options["mou_generator"] = "pso"
    pst.pestpp_options["mou_population_size"] = 30
    pst.pestpp_options["ies_debug_fail_remainder"] = True
    pst.pestpp_options["opt_risk"] = 0.95
    pst.control_data.noptmax = 8
    pst.write(os.path.join(t_d, "zdt1.pst"))
    m1 = os.path.join("mou_tests", "zdt1_test_master_fail_1")
    pyemu.os_utils.start_workers(t_d, exe_path, "zdt1.pst", 35, worker_root="mou_tests",
                                 master_dir=m1, verbose=True,port=port)

    dp_file = os.path.join(m1,"zdt1.0.nested.par_stack.csv")
    dp = pd.read_csv(dp_file,index_col=0)
    op_file = os.path.join(m1,"zdt1.0.nested.obs_stack.csv")
    op = pd.read_csv(dp_file,index_col=0)
    print(dp.shape,op.shape)
    assert dp.shape[0] == op.shape[0]
    
    shutil.copy2(dp_file,os.path.join(t_d,"par_stack.csv"))
    shutil.copy2(op_file,os.path.join(t_d,"obs_stack.csv"))
    pst.pestpp_options["opt_par_stack"] = "par_stack.csv"
    pst.pestpp_options["opt_obs_stack"] = "obs_stack.csv"
    pst.control_data.noptmax = 2
    pst.write(os.path.join(t_d, "zdt1.pst"))
    m1 = os.path.join("mou_tests", "zdt1_test_master_fail_1")
    pyemu.os_utils.start_workers(t_d, exe_path, "zdt1.pst", 35, worker_root="mou_tests",
                                 master_dir=m1, verbose=True,port=port)


    t_d = mou_suite_helper.setup_problem("zdt1", additive_chance=True, risk_obj=False)
    pst = pyemu.Pst(os.path.join(t_d, "zdt1.pst"))
    pst.pestpp_options["opt_chance_points"] = "all"
    pst.pestpp_options["opt_recalc_chance_every"] = 5
    pst.pestpp_options["opt_stack_size"] = 10
    pst.pestpp_options["mou_generator"] = "pso"
    pst.pestpp_options["mou_population_size"] = 30
    pst.pestpp_options["ies_debug_fail_subset"] = True
    pst.pestpp_options["opt_risk"] = 0.95
    pst.control_data.noptmax = 8
    pst.write(os.path.join(t_d, "zdt1.pst"))
    m1 = os.path.join("mou_tests", "zdt1_test_master_fail_1")
    pyemu.os_utils.start_workers(t_d, exe_path, "zdt1.pst", 35, worker_root="mou_tests",
                                 master_dir=m1, verbose=True)


    dp_file = os.path.join(m1,"zdt1.0.nested.par_stack.csv")
    dp = pd.read_csv(dp_file,index_col=0)
    op_file = os.path.join(m1,"zdt1.0.nested.obs_stack.csv")
    op = pd.read_csv(dp_file,index_col=0)
    print(dp.shape,op.shape)
    assert dp.shape[0] == op.shape[0]
    
    shutil.copy2(dp_file,os.path.join(t_d,"par_stack.csv"))
    shutil.copy2(op_file,os.path.join(t_d,"obs_stack.csv"))
    pst.pestpp_options["opt_par_stack"] = "par_stack.csv"
    pst.pestpp_options["opt_obs_stack"] = "obs_stack.csv"
    pst.control_data.noptmax = 2
    pst.write(os.path.join(t_d, "zdt1.pst"))
    m1 = os.path.join("mou_tests", "zdt1_test_master_fail_1")
    pyemu.os_utils.start_workers(t_d, exe_path, "zdt1.pst", 35, worker_root="mou_tests",
                                 master_dir=m1, verbose=True,port=port)

    t_d = mou_suite_helper.setup_problem("zdt1", additive_chance=True,risk_obj=False)
    pst = pyemu.Pst(os.path.join(t_d, "zdt1.pst"))
    pst.pestpp_options["opt_chance_points"] = "single"
    pst.pestpp_options["opt_recalc_chance_every"] = 1000
    pst.pestpp_options["opt_stack_size"] = 10
    pst.pestpp_options["mou_generator"] = "pso"
    pst.pestpp_options["mou_population_size"] = 30
    pst.pestpp_options["ies_debug_fail_subset"] = True
    pst.pestpp_options["opt_risk"] = 0.95
    pst.control_data.noptmax = 8
    pst.write(os.path.join(t_d, "zdt1.pst"))
    m1 = os.path.join("mou_tests", "zdt1_test_master_fail_1")
    pyemu.os_utils.start_workers(t_d, exe_path, "zdt1.pst", 35, worker_root="mou_tests",
                                 master_dir=m1, verbose=True,port=port)



def invest():
    cases = ["kur","sch","srn","tkn","constr","zdt2","zdt3","zdt4","zdt6"]
    noptmax = 100
    for case in cases:
        #t_d = mou_suite_helper.setup_problem("fon")
        mou_suite_helper.run_problem(case,noptmax=noptmax)
        # mou_suite_helper.run_problem_chance(case, noptmax=noptmax,risk_obj=True,chance_points="all",
        #                                     recalc=10000)
        mou_suite_helper.run_problem_chance(case, noptmax=noptmax, risk_obj=True, chance_points="single",
                                            recalc=10000)
        mou_suite_helper.run_problem_chance(case, noptmax=noptmax, risk_obj=False, chance_points="all",
                                            recalc=10000)
        mou_suite_helper.run_problem_chance(case, noptmax=noptmax, risk_obj=False, chance_points="single",
                                            recalc=10000)

def invest_2():
    cases = ["zdt1","zdt2","zdt3","zdt4","zdt6"]
    noptmax = 100
    for case in cases:
        #t_d = mou_suite_helper.setup_problem("fon")

        mou_suite_helper.run_problem(case, noptmax=noptmax, generator="sbx")
        mou_suite_helper.run_problem(case, noptmax=noptmax, generator="de")

        # mou_suite_helper.run_problem_chance(case, noptmax=noptmax,risk_obj=True,chance_points="all",
        #                                     recalc=10000)

def invest_3():
    # cases = ["zdt2", "zdt3", "zdt4", "zdt6", "sch", "srn", "tkn", "constr"]
    # for case in cases:
    #     mou_suite_helper.run_problem(case, generator="sbx", env="spea")
    #     mou_suite_helper.run_problem(case, generator="sbx", env="nsga")
    #     mou_suite_helper.run_problem(case, generator="de,sbx,pm", env="nsga")
    #     mou_suite_helper.run_problem(case, generator="de,sbx,pm", env="spea")
    cases = ["tkn", "constr","zdt1", "zdt3"]
    for case in cases:
        mou_suite_helper.run_problem(case, generator="de", env="spea",self_adaptive=True)
        mou_suite_helper.run_problem(case, generator="de", env="nsga",self_adaptive=True)
        # mou_suite_helper.run_problem(case, generator="pm", env="spea",self_adaptive=True)
        # mou_suite_helper.run_problem(case, generator="pm", env="nsga",self_adaptive=True)
        # mou_suite_helper.run_problem(case, generator="sbx", env="spea",self_adaptive=True)
        # mou_suite_helper.run_problem(case, generator="sbx", env="nsga",self_adaptive=True)
        # mou_suite_helper.run_problem(case, generator="de,sbx,pm", env="nsga",self_adaptive=True)
        # mou_suite_helper.run_problem(case, generator="de,sbx,pm", env="spea",self_adaptive=True)
        # mou_suite_helper.run_problem(case, generator="de", env="spea")
        # mou_suite_helper.run_problem(case, generator="de", env="nsga")
        # mou_suite_helper.run_problem(case, generator="pm", env="spea")
        # mou_suite_helper.run_problem(case, generator="pm", env="nsga")
        # mou_suite_helper.run_problem(case, generator="sbx", env="spea")
        # mou_suite_helper.run_problem(case, generator="sbx", env="nsga")
        # mou_suite_helper.run_problem(case, generator="de,sbx,pm", env="nsga")
        # mou_suite_helper.run_problem(case, generator="de,sbx,pm", env="spea")

def all_infeas_test():
    t_d = mou_suite_helper.setup_problem("tkn")
    pst = pyemu.Pst(os.path.join(t_d,"tkn.pst"))
    obs = pst.observation_data
    print(obs)
    obs.loc["const_1","obsval"] = -1e10
    pst.pestpp_options["mou_population_size"] = 15
    pst.pestpp_options["mou_generator"] = "de"
    pst.pestpp_options["mou_env_selector"] = "spea"
    pst.control_data.noptmax = 2
    pst.write(os.path.join(t_d,"tkn.pst"))
    m1 = os.path.join("mou_tests", "test_master_all_infeas")
    pyemu.os_utils.start_workers(t_d, exe_path, "tkn.pst", 15, worker_root="mou_tests",
                                 master_dir=m1, verbose=True,port=port)


    out_file = os.path.join(m1,"tkn.obs_pop.csv".format(pst.control_data.noptmax))
    assert os.path.exists(out_file)
    df = pd.read_csv(out_file)
    assert df.shape[0] == pst.pestpp_options["mou_population_size"]

    pst.pestpp_options["mou_generator"] = "sbx"
    pst.pestpp_options["mou_env_selector"] = "spea"
    pst.write(os.path.join(t_d,"tkn.pst"))
    m1 = os.path.join("mou_tests", "test_master_all_infeas")
    pyemu.os_utils.start_workers(t_d, exe_path, "tkn.pst", 15, worker_root="mou_tests",
                                 master_dir=m1, verbose=True,port=port)


    out_file = os.path.join(m1,"tkn.obs_pop.csv".format(pst.control_data.noptmax))
    assert os.path.exists(out_file)
    df = pd.read_csv(out_file)
    assert df.shape[0] == pst.pestpp_options["mou_population_size"]

    pst.pestpp_options["mou_generator"] = "de"
    pst.pestpp_options["mou_env_selector"] = "nsga"
    pst.write(os.path.join(t_d,"tkn.pst"))
    m1 = os.path.join("mou_tests", "test_master_all_infeas")
    pyemu.os_utils.start_workers(t_d, exe_path, "tkn.pst", 15, worker_root="mou_tests",
                                 master_dir=m1, verbose=True,port=port)


    out_file = os.path.join(m1,"tkn.obs_pop.csv".format(pst.control_data.noptmax))
    assert os.path.exists(out_file)
    df = pd.read_csv(out_file)
    assert df.shape[0] == pst.pestpp_options["mou_population_size"]

    pst.pestpp_options["mou_generator"] = "pso"
    pst.pestpp_options["mou_env_selector"] = "nsga"
    pst.write(os.path.join(t_d,"tkn.pst"))
    m1 = os.path.join("mou_tests", "test_master_all_infeas")
    pyemu.os_utils.start_workers(t_d, exe_path, "tkn.pst", 15, worker_root="mou_tests",
                                 master_dir=m1, verbose=True,port=port)


    out_file = os.path.join(m1,"tkn.obs_pop.csv".format(pst.control_data.noptmax))
    assert os.path.exists(out_file)
    df = pd.read_csv(out_file)
    assert df.shape[0] == pst.pestpp_options["mou_population_size"]

def invest_4():
    #mou_suite_helper.run_problem(test_case="zdt1", pop_size=100, noptmax=100, generator="de", env="nsga", self_adaptive=False)
    #mou_suite_helper.run_problem(test_case="zdt1", pop_size=100, noptmax=100, generator="de", env="nsga",
    #                             self_adaptive=True)
    mou_suite_helper.plot_results(os.path.join("mou_tests","zdt1_master_generator=de_env=nsga_popsize=100_risk=0.5_riskobj=False_adaptive=False"))
    mou_suite_helper.plot_results(os.path.join("mou_tests",
                                               "zdt1_master_generator=de_env=nsga_popsize=100_risk=0.5_riskobj=False_adaptive=True"))


def restart_dv_test():

    t_d = mou_suite_helper.setup_problem("tkn")
    pst = pyemu.Pst(os.path.join(t_d, "tkn.pst"))
    obs = pst.observation_data
    print(obs)
    obs.loc["const_1", "obsval"] = -1e10
    pst.pestpp_options["mou_population_size"] = 15
    pst.pestpp_options["mou_generator"] = "de"
    pst.pestpp_options["mou_env_selector"] = "spea"
    pst.control_data.noptmax = 2
    pst.write(os.path.join(t_d, "tkn.pst"))
    pyemu.os_utils.run("{0} tkn.pst".format(exe_path), cwd=t_d)
    shutil.copy2(os.path.join(t_d,"tkn.dv_pop.csv".format(pst.control_data.noptmax)),os.path.join(t_d,"restart.csv"))
    pst.pestpp_options["mou_dv_population_file"] = "restart.csv"
    pst.control_data.noptmax = 1
    pst.write(os.path.join(t_d, "tkn.pst"))
    pyemu.os_utils.run("{0} tkn.pst".format(exe_path), cwd=t_d)
    df = pd.read_csv(os.path.join(t_d,"tkn.dv_pop.csv".format(pst.control_data.noptmax)))
    gen_num = df.real_name.apply(lambda x: int(x.split("=")[1].split('_')[0]))
    print(gen_num.max())
    assert gen_num.max() == 3

def chance_all_binary_test():

    t_d = mou_suite_helper.setup_problem("constr", additive_chance=True, risk_obj=False)
    pst = pyemu.Pst(os.path.join(t_d, "constr.pst"))
    pst.pestpp_options["opt_chance_points"] = "all"
    pst.pestpp_options["opt_recalc_chance_every"] = 5
    pst.pestpp_options["opt_stack_size"] = 4
    pst.pestpp_options["mou_generator"] = "de"
    pst.pestpp_options["mou_population_size"] = 5
    pst.pestpp_options["opt_risk"] = 0.95
    pst.pestpp_options["save_binary"] = True
    par = pst.parameter_data
    par.loc["dv_1","partrans"] = "fixed"
    par.loc["obj1_add_par","partrans"] = "fixed"
    pst.control_data.noptmax = 1
    pst.write(os.path.join(t_d, "constr.pst"))
    m1 = os.path.join("mou_tests", "constr_test_master_chance_binary")
    pyemu.os_utils.start_workers(t_d, exe_path, "constr.pst", 35, worker_root="mou_tests",
                                 master_dir=m1, verbose=True,port=port)
    pe = pyemu.ParameterEnsemble.from_binary(pst=pst,filename=os.path.join(m1,"constr.0.nested.par_stack.jcb"))
    oe = pyemu.ObservationEnsemble.from_binary(pst=pst, filename=os.path.join(m1, "constr.0.nested.obs_stack.jcb"))
    dva = pyemu.ParameterEnsemble.from_binary(pst=pst,filename=os.path.join(m1,"constr.archive.dv_pop.jcb"))
    dv = pyemu.ParameterEnsemble.from_binary(pst=pst, filename=os.path.join(m1, "constr.dv_pop.jcb"))
    opa = pyemu.ObservationEnsemble.from_binary(pst=pst, filename=os.path.join(m1, "constr.archive.obs_pop.jcb"))
    op = pyemu.ObservationEnsemble.from_binary(pst=pst, filename=os.path.join(m1, "constr.obs_pop.jcb"))

    shutil.copy(os.path.join(m1,"constr.0.nested.par_stack.jcb"),os.path.join(t_d,"par_stack.jcb"))
    shutil.copy(os.path.join(m1, "constr.0.nested.obs_stack.jcb"), os.path.join(t_d, "obs_stack.jcb"))
    shutil.copy(os.path.join(m1, "constr.0.dv_pop.jcb"), os.path.join(t_d, "dv_pop.jcb"))
    pst.pestpp_options["opt_par_stack"] = "par_stack.jcb"
    pst.write(os.path.join(t_d, "constr.pst"))
    m2 = os.path.join("mou_tests", "constr_test_master_chance_binary_restart")
    pyemu.os_utils.start_workers(t_d, exe_path, "constr.pst", 35, worker_root="mou_tests",
                                 master_dir=m2, verbose=True, port=port)
    pe = pyemu.ParameterEnsemble.from_binary(pst=pst, filename=os.path.join(m2, "constr.0.nested.par_stack.jcb"))
    oe = pyemu.ObservationEnsemble.from_binary(pst=pst, filename=os.path.join(m2, "constr.0.nested.obs_stack.jcb"))
    dva = pyemu.ParameterEnsemble.from_binary(pst=pst,filename=os.path.join(m2,"constr.archive.dv_pop.jcb"))
    dv = pyemu.ParameterEnsemble.from_binary(pst=pst, filename=os.path.join(m2, "constr.dv_pop.jcb"))
    opa = pyemu.ObservationEnsemble.from_binary(pst=pst, filename=os.path.join(m2, "constr.archive.obs_pop.jcb"))
    op = pyemu.ObservationEnsemble.from_binary(pst=pst, filename=os.path.join(m2, "constr.obs_pop.jcb"))

    shutil.copy(os.path.join(m1, "constr.0.nested.par_stack.jcb"), os.path.join(t_d, "par_stack.jcb"))
    shutil.copy(os.path.join(m1, "constr.0.nested.obs_stack.jcb"), os.path.join(t_d, "obs_stack.jcb"))
    shutil.copy(os.path.join(m1, "constr.0.dv_pop.jcb"), os.path.join(t_d, "dv_pop.jcb"))
    pst.pestpp_options["opt_par_stack"] = "par_stack.jcb"
    pst.write(os.path.join(t_d, "constr.pst"))
    m2 = os.path.join("mou_tests", "constr_test_master_chance_binary_restart")
    pyemu.os_utils.start_workers(t_d, exe_path, "constr.pst", 35, worker_root="mou_tests",
                                 master_dir=m2, verbose=True, port=port)
    pe = pyemu.ParameterEnsemble.from_binary(pst=pst, filename=os.path.join(m2, "constr.0.nested.par_stack.jcb"))
    oe = pyemu.ObservationEnsemble.from_binary(pst=pst, filename=os.path.join(m2, "constr.0.nested.obs_stack.jcb"))
    dva = pyemu.ParameterEnsemble.from_binary(pst=pst, filename=os.path.join(m2, "constr.archive.dv_pop.jcb"))
    dv = pyemu.ParameterEnsemble.from_binary(pst=pst, filename=os.path.join(m2, "constr.dv_pop.jcb"))
    opa = pyemu.ObservationEnsemble.from_binary(pst=pst, filename=os.path.join(m2, "constr.archive.obs_pop.jcb"))
    op = pyemu.ObservationEnsemble.from_binary(pst=pst, filename=os.path.join(m2, "constr.obs_pop.jcb"))



def risk_demo(case="zdt1",noptmax=100,std_weight=0.05,mou_gen="de",pop_size=100,num_workers=30):

    obj_names = ["obj_1"]
    if "zdt" in case:
        obj_names.append("obj_2")
    constr_bnd = std_weight * 2

    # spec'd risk = 0.95, stack, reuse
    t_d = mou_suite_helper.setup_problem(case, additive_chance=True, risk_obj=False)
    pst = pyemu.Pst(os.path.join(t_d, case + ".pst"))
    par = pst.parameter_data
    add_par = par.loc[par.parnme.str.contains("_add_"), "parnme"]
    if case == "constr":
        par.loc[add_par, "parubnd"] = constr_bnd
        par.loc[add_par, "parlbnd"] = -constr_bnd

    pst.pestpp_options["opt_chance_points"] = "single"
    pst.pestpp_options["opt_recalc_chance_every"] = 10000
    pst.pestpp_options["opt_stack_size"] = 100
    pst.pestpp_options["mou_generator"] = mou_gen
    pst.pestpp_options["mou_population_size"] = pop_size
    pst.pestpp_options["opt_risk"] = 0.95
    pst.pestpp_options["save_binary"] = True
    pst.observation_data.loc[pst.nnz_obs_names, "weight"] = std_weight
    pst.pestpp_options["opt_std_weights"] = False
    pst.control_data.noptmax = noptmax
    pst.write(os.path.join(t_d, case + ".pst"))
    m1 = os.path.join("mou_tests", case + "_test_master_95_singlept_match_" + mou_gen)
    pyemu.os_utils.start_workers(t_d, exe_path, case + ".pst", num_workers, worker_root="mou_tests",
                                 master_dir=m1, verbose=True, port=port)

    # spec'd risk = 0.05, stack, reuse
    t_d = mou_suite_helper.setup_problem(case, additive_chance=True, risk_obj=False)
    pst = pyemu.Pst(os.path.join(t_d, case + ".pst"))
    par = pst.parameter_data
    add_par = par.loc[par.parnme.str.contains("_add_"), "parnme"]
    if case == "constr":
        par.loc[add_par, "parubnd"] = constr_bnd
        par.loc[add_par, "parlbnd"] = -constr_bnd

    pst.pestpp_options["opt_chance_points"] = "single"
    pst.pestpp_options["opt_recalc_chance_every"] = 10000
    pst.pestpp_options["opt_stack_size"] = 100
    pst.pestpp_options["mou_generator"] = mou_gen
    pst.pestpp_options["mou_population_size"] = pop_size
    pst.pestpp_options["opt_risk"] = 0.05
    pst.pestpp_options["save_binary"] = True
    pst.observation_data.loc[pst.nnz_obs_names, "weight"] = std_weight
    pst.pestpp_options["opt_std_weights"] = False
    pst.control_data.noptmax = noptmax
    pst.write(os.path.join(t_d, case + ".pst"))
    m1 = os.path.join("mou_tests", case + "_test_master_05_singlept_match_" + mou_gen)
    pyemu.os_utils.start_workers(t_d, exe_path, case + ".pst", num_workers, worker_root="mou_tests",
                                 master_dir=m1, verbose=True, port=port)

    # risk obj, stack, all chance pts, reuse
    t_d = mou_suite_helper.setup_problem(case, additive_chance=True, risk_obj=True)
    pst = pyemu.Pst(os.path.join(t_d, case+".pst"))
    par = pst.parameter_data
    add_par = par.loc[par.parnme.str.contains("_add_"),"parnme"]
    if case == "constr":
        par.loc[add_par, "parubnd"] = constr_bnd
        par.loc[add_par, "parlbnd"] = -constr_bnd

    pst.pestpp_options["opt_chance_points"] = "all"
    pst.pestpp_options["opt_recalc_chance_every"] = 10000
    pst.pestpp_options["opt_stack_size"] = 100
    pst.pestpp_options["mou_generator"] = mou_gen
    pst.pestpp_options["mou_population_size"] = pop_size
    pst.pestpp_options["opt_risk"] = 0.95
    pst.pestpp_options["save_binary"] = True
    pst.observation_data.loc[pst.nnz_obs_names,"weight"] = std_weight
    pst.pestpp_options["opt_std_weights"] = False
    pst.control_data.noptmax = noptmax * 3
    pst.write(os.path.join(t_d, case+".pst"))
    m1 = os.path.join("mou_tests", case+"_test_master_riskobj_allpts_more_"+mou_gen)
    pyemu.os_utils.start_workers(t_d, exe_path, case+".pst", num_workers, worker_root="mou_tests",
                                 master_dir=m1, verbose=True, port=port)

    # risk obj, stack, single pt, reuse
    t_d = mou_suite_helper.setup_problem(case, additive_chance=True, risk_obj=True)
    pst = pyemu.Pst(os.path.join(t_d, case+".pst"))
    par = pst.parameter_data
    add_par = par.loc[par.parnme.str.contains("_add_"), "parnme"]
    if case == "constr":
        par.loc[add_par, "parubnd"] = constr_bnd
        par.loc[add_par, "parlbnd"] = -constr_bnd

    pst.pestpp_options["opt_chance_points"] = "single"
    pst.pestpp_options["opt_recalc_chance_every"] = 10000
    pst.pestpp_options["opt_stack_size"] = 100
    pst.pestpp_options["mou_generator"] = mou_gen
    pst.pestpp_options["mou_population_size"] = pop_size
    pst.pestpp_options["opt_risk"] = 0.95
    pst.pestpp_options["save_binary"] = True
    pst.observation_data.loc[pst.nnz_obs_names,"weight"] = std_weight
    pst.pestpp_options["opt_std_weights"] = False
    pst.control_data.noptmax = noptmax * 3
    pst.write(os.path.join(t_d, case+".pst"))
    m1 = os.path.join("mou_tests", case+"_test_master_riskobj_singlept_more_"+mou_gen)
    pyemu.os_utils.start_workers(t_d, exe_path, case+".pst", num_workers, worker_root="mou_tests",
                                 master_dir=m1, verbose=True, port=port)

    # deterministic
    t_d = mou_suite_helper.setup_problem(case, additive_chance=False, risk_obj=False)
    pst = pyemu.Pst(os.path.join(t_d, case+".pst"))
    par = pst.parameter_data
    add_par = par.loc[par.parnme.str.contains("_add_"), "parnme"]
    if case == "constr":
        par.loc[add_par, "parubnd"] = constr_bnd
        par.loc[add_par, "parlbnd"] = -constr_bnd

    pst.pestpp_options["mou_generator"] = mou_gen
    pst.pestpp_options["mou_population_size"] = pop_size
    pst.pestpp_options["opt_risk"] = 0.5
    pst.pestpp_options["save_binary"] = True
    pst.observation_data.loc[pst.nnz_obs_names,"weight"] = std_weight
    pst.control_data.noptmax = noptmax
    pst.write(os.path.join(t_d, case+".pst"))
    m1 = os.path.join("mou_tests", case+"_test_master_deter_"+mou_gen)
    pyemu.os_utils.start_workers(t_d, exe_path, case+".pst", num_workers, worker_root="mou_tests",
                                 master_dir=m1, verbose=True, port=port)


    # spec'd risk = 0.95, std weights
    t_d = mou_suite_helper.setup_problem(case, additive_chance=True, risk_obj=False)
    pst = pyemu.Pst(os.path.join(t_d, case+".pst"))
    par = pst.parameter_data
    add_par = par.loc[par.parnme.str.contains("_add_"), "parnme"]
    if case == "constr":
        par.loc[add_par, "parubnd"] = constr_bnd
        par.loc[add_par, "parlbnd"] = -constr_bnd

    pst.observation_data.loc[pst.nnz_obs_names,"weight"] = std_weight
    pst.pestpp_options["opt_std_weights"] = True
    pst.pestpp_options["mou_generator"] = mou_gen
    pst.pestpp_options["mou_population_size"] = pop_size
    pst.pestpp_options["opt_risk"] = 0.95
    pst.pestpp_options["save_binary"] = True
    pst.control_data.noptmax = noptmax
    pst.write(os.path.join(t_d, case+".pst"))
    m2 = os.path.join("mou_tests", case+"_test_master_95_"+mou_gen)
    pyemu.os_utils.start_workers(t_d, exe_path, case+".pst", num_workers, worker_root="mou_tests",
                                 master_dir=m2, verbose=True, port=port)


    # spec's risk = 0.05, std weights,
    t_d = mou_suite_helper.setup_problem(case, additive_chance=True, risk_obj=False)
    pst = pyemu.Pst(os.path.join(t_d, case+".pst"))
    par = pst.parameter_data
    add_par = par.loc[par.parnme.str.contains("_add_"), "parnme"]
    if case == "constr":
        par.loc[add_par, "parubnd"] = constr_bnd
        par.loc[add_par, "parlbnd"] = -constr_bnd

    pst.observation_data.loc[pst.nnz_obs_names,"weight"] = std_weight
    pst.pestpp_options["opt_std_weights"] = True
    pst.pestpp_options["mou_generator"] = mou_gen
    pst.pestpp_options["mou_population_size"] = pop_size
    pst.pestpp_options["opt_risk"] = 0.05
    pst.pestpp_options["save_binary"] = True
    pst.control_data.noptmax = noptmax
    pst.write(os.path.join(t_d, case+".pst"))
    m3 = os.path.join("mou_tests", case+"_test_master_05_"+mou_gen)
    pyemu.os_utils.start_workers(t_d, exe_path, case+".pst", num_workers, worker_root="mou_tests",
                                 master_dir=m3, verbose=True, port=port)

    # risk obj, std weights
    t_d = mou_suite_helper.setup_problem(case, additive_chance=True, risk_obj=True)
    pst = pyemu.Pst(os.path.join(t_d, case+".pst"))
    par = pst.parameter_data
    add_par = par.loc[par.parnme.str.contains("_add_"), "parnme"]
    if case == "constr":
        par.loc[add_par, "parubnd"] = constr_bnd
        par.loc[add_par, "parlbnd"] = -constr_bnd

    pst.observation_data.loc[pst.nnz_obs_names,"weight"] = std_weight
    pst.pestpp_options["opt_std_weights"] = True
    pst.pestpp_options["mou_generator"] = mou_gen
    pst.pestpp_options["mou_population_size"] = pop_size
    pst.pestpp_options["opt_risk"] = 0.05
    pst.pestpp_options["save_binary"] = True
    pst.control_data.noptmax = noptmax
    pst.write(os.path.join(t_d, case+".pst"))
    m4 = os.path.join("mou_tests", case+"_test_master_riskobj_match_"+mou_gen)
    pyemu.os_utils.start_workers(t_d, exe_path, case+".pst", num_workers, worker_root="mou_tests",
                                 master_dir=m4, verbose=True, port=port)

    # risk obj ,std weights
    t_d = mou_suite_helper.setup_problem(case, additive_chance=True, risk_obj=True)
    pst = pyemu.Pst(os.path.join(t_d, case+".pst"))
    par = pst.parameter_data
    add_par = par.loc[par.parnme.str.contains("_add_"), "parnme"]
    if case == "constr":
        par.loc[add_par, "parubnd"] = constr_bnd
        par.loc[add_par, "parlbnd"] = -constr_bnd

    pst.observation_data.loc[pst.nnz_obs_names,"weight"] = std_weight
    pst.pestpp_options["opt_std_weights"] = True
    pst.pestpp_options["mou_generator"] = mou_gen
    pst.pestpp_options["mou_population_size"] = pop_size
    pst.pestpp_options["opt_risk"] = 0.05
    pst.pestpp_options["save_binary"] = True
    pst.control_data.noptmax = noptmax * 3
    pst.write(os.path.join(t_d, case+".pst"))
    m5 = os.path.join("mou_tests", case+"_test_master_riskobj_more_"+mou_gen)
    pyemu.os_utils.start_workers(t_d, exe_path, case+".pst", num_workers, worker_root="mou_tests",
                                 master_dir=m5, verbose=True, port=port)


    # risk obj, stack, every
    t_d = mou_suite_helper.setup_problem(case, additive_chance=True, risk_obj=True)
    pst = pyemu.Pst(os.path.join(t_d, case + ".pst"))
    par = pst.parameter_data
    add_par = par.loc[par.parnme.str.contains("_add_"), "parnme"]
    if case == "constr":
        par.loc[add_par, "parubnd"] = constr_bnd
        par.loc[add_par, "parlbnd"] = -constr_bnd

    pst.observation_data.loc[pst.nnz_obs_names, "weight"] = std_weight
    pst.pestpp_options["opt_std_weights"] = True
    pst.pestpp_options["mou_generator"] = mou_gen
    pst.pestpp_options["mou_population_size"] = pop_size
    pst.pestpp_options["opt_risk"] = 0.05
    pst.pestpp_options["save_binary"] = True
    pst.pestpp_options["opt_chance_points"] = "all"
    pst.pestpp_options["opt_recalc_chance_every"] = 1
    pst.pestpp_options["opt_stack_size"] = 100
    pst.control_data.noptmax = noptmax * 3
    pst.write(os.path.join(t_d, case + ".pst"))
    m5 = os.path.join("mou_tests", case + "_test_master_allpts_every_riskobj_more_" + mou_gen)
    pyemu.os_utils.start_workers(t_d, exe_path, case + ".pst", num_workers, worker_root="mou_tests",
                                 master_dir=m5, verbose=True, port=port)


def plot_risk_demo_multi(case = "zdt1", mou_gen="de"):
    import matplotlib.pyplot as plt
    m_deter = os.path.join("mou_tests",case+"_test_master_deter_"+mou_gen)
    m_ravr = os.path.join("mou_tests",case+"_test_master_95_"+mou_gen)
    m_rtol = os.path.join("mou_tests", case+"_test_master_05_"+mou_gen)
    m_robj = os.path.join("mou_tests",case+"_test_master_riskobj_match_"+mou_gen)
    m_robjm = os.path.join("mou_tests", case+"_test_master_riskobj_more_"+mou_gen)

    fig, ax = plt.subplots(1,1,figsize=(10,10))
    for d,c in zip([m_deter,m_ravr,m_rtol,m_robjm],['g','b','r',None,None]):

        pst = pyemu.Pst(os.path.join(d,case+".pst"))
        df = pd.read_csv(os.path.join(d,case+".pareto.archive.summary.csv"))
        mxgen = df.generation.max()
        
        print(d,mxgen)
        df = df.loc[df.generation==mxgen,:]
        print(d,mxgen)
        if "riskobj" in d:
            #print(df.head().loc[:,['obj_1',"obj_2",'_risk_']])
            #ax.scatter(df.obj_1.values[:2],df.obj_2.values[:2],marker="+",c=df._risk_[:2],cmap='jet')
            rdf = df#.loc[df._risk_ < 0.05,:]
            ax.scatter(rdf.obj_1,rdf.obj_2,marker="o",c=1 - rdf._risk_.values,cmap='jet')
        else:
            ax.scatter(df.obj_1.values,df.obj_2.values,marker='.',color=c)

    if case == "zdt1":
        x0 = np.linspace(0,1,1000)
        o1,o2 = [],[]
        for xx0 in x0:
            x = np.zeros(30)
            x[0] = xx0
            ret_vals = mou_suite_helper.zdt1(x)
            o1.append(ret_vals[0][0])
            o2.append(ret_vals[0][1])

        ax.plot(o1,o2,"k",label="truth")
    plt.show()


def plot_risk_demo_multi_3pane(case="zdt1",mou_gen="de"):
    import matplotlib.pyplot as plt
    #m_d = os.path.join("mou_tests", case + "_test_master_riskobj_more_"+mou_gen)
    m_d = os.path.join("mou_tests", case + "_test_master_95_" + mou_gen)
    pst = pyemu.Pst(os.path.join(m_d, case + ".pst"))
    obj_names = pst.pestpp_options["mou_objectives"].lower().split(',')
    df = pd.read_csv(os.path.join(m_d, case + ".pareto.archive.summary.csv"))
    mxgen = df.generation.max()
    print(mxgen)
    df = df.loc[df.generation == mxgen, :]
    df = df.loc[df.nsga2_front==1,:]

    #df = df.loc[df.obj_2<1,:]
    
    #df = pyemu.ObservationEnsemble.from_binary(pst=pst,filename=os.path.join(m_d,case+".archive.obs_pop.jcb"))
    #pe = pyemu.ParameterEnsemble.from_binary(pst=pst,filename=os.path.join(m_d,case+".archive.dv_pop.jcb"))
    #print(pe.loc[:,"dv_1"].min())
    #return
    #df.loc[:,"_risk_"] = pe.loc[df.index,"_risk_"].values
    #df = df.loc[df._risk_.apply(lambda x: x > 0.95),:]
    rvals = None
    if "_risk_" in obj_names:
        rvals = df.loc[:,"_risk_"].values.copy()

    fig, axes = plt.subplots(len(obj_names), len(obj_names), figsize=(10, 10))
    for i,o1 in enumerate(obj_names):
        for j in range(i+1):
            o2 = obj_names[j]
            ax = axes[i, j]
            v1 = df.loc[:, o1].values
            v2 = df.loc[:, o2].values
            if i == j:
                ax.hist(v2,bins=20,facecolor="0.5",edgecolor="none",alpha=0.5)
                ax.set_xlabel(o1)
                ax.set_yticks([])
            else:
                if rvals is None:
                    ax.scatter(v2,v1,marker=".",color="0.5",s=20)
                else:
                    ax.scatter(v2,v1,marker=".",c=rvals,s=20)
                ax.set_xlabel(o2)
                ax.set_ylabel(o1)

            if case == "zdt1" and o1 == "obj_2" and o2 == "obj_1":
                x0 = np.linspace(0, 1, 1000)
                ov1, ov2 = [], []
                for xx0 in x0:
                    x = np.zeros(30)
                    x[0] = xx0
                    ret_vals = mou_suite_helper.zdt1(x)
                    ov1.append(ret_vals[0][0])
                    ov2.append(ret_vals[0][1])

                ax.plot(ov1, ov2, "k", label="truth")

        for j in range(i+1,len(obj_names)):
            o2 = obj_names[j]
            ax = axes[i, j]
            v1 = df.loc[:, o1].values
            v2 = df.loc[:, o2].values
            if rvals is None:
                ax.scatter(v2,v1,marker=".",color="0.5",s=20)
            else:
                ax.scatter(v2,v1,marker=".",c=rvals,s=20)
            ax.set_xlabel(o2)
            ax.set_ylabel(o1)


            if case == "zdt1" and o1 == "obj_1" and o2 == "obj_2":
                x0 = np.linspace(0, 1, 1000)
                ov1, ov2 = [], []
                for xx0 in x0:
                    x = np.zeros(30)
                    x[0] = xx0
                    ret_vals = mou_suite_helper.zdt1(x)
                    ov1.append(ret_vals[0][0])
                    ov2.append(ret_vals[0][1])

                ax.plot(ov2, ov1, "k", label="truth")
    plt.show()


def plot_risk_demo_rosen():
    case = "rosenc"
    import matplotlib.pyplot as plt
    mou_gen = "de"
    m_deter = os.path.join("mou_tests",case+"_test_master_deter_"+mou_gen)
    m_ravr = os.path.join("mou_tests",case+"_test_master_95_"+mou_gen)
    m_rtol = os.path.join("mou_tests", case+"_test_master_05_"+mou_gen)
    #m_robj = os.path.join("mou_tests",case+"_test_master_riskobj_match")
    m_robjm = os.path.join("mou_tests", case+"_test_master_riskobj_match_"+mou_gen)
    bins = 30#np.linspace(-5,5,30)
    fig, axes = plt.subplots(1,2,figsize=(8,4))
    #axes = axes.flatten()
    for d,c,label in zip([m_deter,m_ravr,m_rtol,m_robjm],['g','b','r',"c","m"],["risk neutral (risk=0.5)","risk averse (risk=0.95)","risk tolerant (risk=0.05)","risk as an objective"]):

        pst = pyemu.Pst(os.path.join(d,case+".pst"))
        df = pd.read_csv(os.path.join(d,case+".pareto.archive.summary.csv"))
        df2 = pd.read_csv(os.path.join(d[:-2] + "pso",case+".pareto.archive.summary.csv"))
        mxgen = df.generation.max()
        #mxgen = 10
        #print(d,mxgen)
        df = df.loc[df.generation==mxgen,:]
        mxgen2 = df2.generation.max()
        df2 = df2.loc[df2.generation==mxgen2,:]
        print(d,mxgen)
        ax = axes[0]

        if "riskobj" in d:
            #print(df.head().loc[:,['obj_1',"obj_2",'_risk_']])
            #ax.scatter(df.obj_1.values[:2],df.obj_2.values[:2],marker="+",c=df._risk_[:2],cmap='jet')
            rdf = df#.loc[df._risk_ < 0.05,:]
            #axt = plt.twinx(ax)
            ax.scatter(rdf.obj_1,rdf._risk_,marker="o",c=1 - rdf._risk_.values,cmap='jet',label="risk vs objective function\npareto frontier")
            ax.set_ylabel("risk")


        #ax.hist(df.obj_1.values,bins=bins,facecolor="0.5",edgecolor="none",alpha=0.5,density=False)

        #ax.hist(df.obj_1.values, bins=bins, facecolor=0.5, density=False)

        ax.plot([df.obj_1.mean(),df.obj_1.mean()],[0,1],color=c,ls="--",label=label)
        #ax.set_xlim(bins.min(),bins.max())
        ax.set_xlim(-5,5)
        ax.set_title("A) DE",loc="left")
        ax.set_xlabel("objective function value")

        ax = axes[1]

        if "riskobj" in d:
            # print(df.head().loc[:,['obj_1',"obj_2",'_risk_']])
            # ax.scatter(df.obj_1.values[:2],df.obj_2.values[:2],marker="+",c=df._risk_[:2],cmap='jet')
            rdf = df2 # .loc[df._risk_ < 0.05,:]
            # axt = plt.twinx(ax)
            ax.scatter(rdf.obj_1, rdf._risk_, marker="o", c=1 - rdf._risk_.values, cmap='jet',
                       label="risk vs objective function\npareto frontier")
            ax.set_ylabel("risk")

        # ax.hist(df.obj_1.values,bins=bins,facecolor="0.5",edgecolor="none",alpha=0.5,density=False)

        # ax.hist(df.obj_1.values, bins=bins, facecolor=0.5, density=False)

        ax.plot([df2.obj_1.mean(), df2.obj_1.mean()], [0, 1], color=c, ls="--", label=label)
        # ax.set_xlim(bins.min(),bins.max())
        ax.set_xlim(-5, 5)
        ax.set_title("B) PSO", loc="left")
        ax.set_xlabel("objective function value")
        #ax.set_yticks([])

    for ax in axes:
        ax.set_ylim(0,1)
        ax.legend(loc="upper left")

    #plt.show()
    plt.savefig("rosenc_risk_demo.pdf")

def risk_obj_test():
    t_d = mou_suite_helper.setup_problem("constr", additive_chance=True, risk_obj=True)

    pst = pyemu.Pst(os.path.join(t_d, "constr.pst"))
    pst.pestpp_options["opt_std_weights"] = True
    pst.pestpp_options["opt_recalc_chance_every"] = 5
    pst.pestpp_options["mou_generator"] = "de"
    pst.pestpp_options["mou_population_size"] = 5
    pst.pestpp_options["opt_risk"] = 0.05
    pst.pestpp_options["mou_risk_objective"] = True
    pst.observation_data.loc["obj_1","weight"] = 0.5
    pst.observation_data.loc["obj_2", "weight"] = 0.5
    pst.control_data.noptmax = -1
    pst.write(os.path.join(t_d, "constr.pst"))
    m1 = os.path.join("mou_tests", "constr_riskobj_test_master")
    pyemu.os_utils.start_workers(t_d, exe_path, "constr.pst", 5, worker_root="mou_tests",
                                 master_dir=m1, verbose=True, port=port)
    return

    pst = pyemu.Pst(os.path.join(t_d, "constr.pst"))
    pst.pestpp_options["opt_std_weights"] = False
    pst.pestpp_options["opt_chance_points"] = "all"
    pst.pestpp_options["opt_recalc_chance_every"] = 5
    pst.pestpp_options["opt_stack_size"] = 5
    pst.pestpp_options["mou_generator"] = "de"
    pst.pestpp_options["mou_population_size"] = 5
    pst.pestpp_options["opt_risk"] = 0.05
    pst.pestpp_options["mou_risk_objective"] = True
    pst.control_data.noptmax = -1
    pst.write(os.path.join(t_d, "constr.pst"))
    m1 = os.path.join("mou_tests", "constr_riskobj_test_master")
    pyemu.os_utils.start_workers(t_d, exe_path, "constr.pst", 5, worker_root="mou_tests",
                                 master_dir=m1, verbose=True,port=port)
  
    dv = pd.read_csv(os.path.join(m1,"constr.0.dv_pop.csv"),index_col=0)
    dv.loc[:,"_risk_"] = pst.pestpp_options["opt_risk"]
    dv.to_csv(os.path.join(t_d,"dv.csv"))
    pst.pestpp_options["mou_dv_population_file"] = "dv.csv"
    pst.write(os.path.join(t_d, "constr.pst"))
    m1 = os.path.join("mou_tests", "constr_riskobj_test_master")
    pyemu.os_utils.start_workers(t_d, exe_path, "constr.pst", 5, worker_root="mou_tests",
                                 master_dir=m1, verbose=True, port=port)
    pst.pestpp_options["mou_risk_objective"] = False
    pst.write(os.path.join(t_d, "constr.pst"))
    m2 = os.path.join("mou_tests", "constr_riskobj_test_master2")
    pyemu.os_utils.start_workers(t_d, exe_path, "constr.pst", 5, worker_root="mou_tests",
                                 master_dir=m2, verbose=True, port=port)

    op1 = pd.read_csv(os.path.join(m1, "constr.0.obs_pop.csv"), index_col=0)
    op2 = pd.read_csv(os.path.join(m2, "constr.0.obs_pop.csv"), index_col=0)
    d = (op1 - op2).apply(np.abs)
    print(d.max())
    assert d.max().max() < 1.0e-10,d.max().max()
    op1 = pd.read_csv(os.path.join(m1, "constr.0.obs_pop.chance.csv"), index_col=0)
    op2 = pd.read_csv(os.path.join(m2, "constr.0.obs_pop.chance.csv"), index_col=0)
    d = (op1 - op2).apply(np.abs)
    print(d.max())
    assert d.max().max() < 1.0e-10, d.max().max()

    shutil.copy2(os.path.join(m2,"constr.0.nested.par_stack.csv"),os.path.join(t_d,"par_stack.csv"))
    shutil.copy2(os.path.join(m2, "constr.0.nested.obs_stack.csv"), os.path.join(t_d, "obs_stack.csv"))
    pst.pestpp_options["opt_par_stack"] = "par_stack.csv"
    pst.pestpp_options["opt_obs_stack"] = "obs_stack.csv"
    dv = pyemu.ParameterEnsemble.from_uniform_draw(pst,5)
    dv.loc[:,"_risk_"] = pst.pestpp_options["opt_risk"]
    pst.pestpp_options["mou_risk_objective"] = True
    dv.to_csv(os.path.join(t_d, "dv.csv"))
    pst.write(os.path.join(t_d, "constr.pst"))
    m1 = os.path.join("mou_tests", "constr_riskobj_test_master")
    pyemu.os_utils.start_workers(t_d, exe_path, "constr.pst", 5, worker_root="mou_tests",
                                 master_dir=m1, verbose=True, port=port)
    pst.pestpp_options["mou_risk_objective"] = False
    pst.write(os.path.join(t_d, "constr.pst"))
    m2 = os.path.join("mou_tests", "constr_riskobj_test_master2")
    pyemu.os_utils.start_workers(t_d, exe_path, "constr.pst", 5, worker_root="mou_tests",
                                 master_dir=m2, verbose=True, port=port)

    op1 = pd.read_csv(os.path.join(m1, "constr.0.obs_pop.csv"), index_col=0)
    op2 = pd.read_csv(os.path.join(m2, "constr.0.obs_pop.csv"), index_col=0)
    d = (op1 - op2).apply(np.abs)
    print(d.max())
    assert d.max().max() < 1.0e-10, d.max().max()
    op1 = pd.read_csv(os.path.join(m1, "constr.0.obs_pop.chance.csv"), index_col=0)
    op2 = pd.read_csv(os.path.join(m2, "constr.0.obs_pop.chance.csv"), index_col=0)
    d = (op1 - op2).apply(np.abs)
    print(d.max())
    assert d.max().max() < 1.0e-10, d.max().max()


def basic_pso_test(case="zdt1"):
    t_d = mou_suite_helper.setup_problem(case, additive_chance=True, risk_obj=False)
    pst = pyemu.Pst(os.path.join(t_d, case+".pst"))
    pst.pestpp_options["mou_generator"] = "pso"
    pst.pestpp_options["opt_risk"] = 0.95
    pst.control_data.noptmax = 3
    pst.write(os.path.join(t_d, case+".pst"))
    m_d = os.path.join("mou_tests", case+"_pso_master_risk")
    pyemu.os_utils.start_workers(t_d, exe_path,  case+".pst", 20, worker_root="mou_tests",
                                 master_dir=m_d, verbose=True, port=port)
    assert os.path.exists(os.path.join(m_d,"{0}.{1}.fosm.jcb".format(case,pst.control_data.noptmax)))

    pst.pestpp_options["mou_generator"] = "de"
    pst.pestpp_options["opt_risk"] = 0.95
    pst.control_data.noptmax = 3
    pst.write(os.path.join(t_d, case+".pst"))
    m_d = os.path.join("mou_tests", case+"_de_master_risk")
    pyemu.os_utils.start_workers(t_d, exe_path, case+".pst", 20, worker_root="mou_tests",
                                 master_dir=m_d, verbose=True, port=port)
    assert os.path.exists(os.path.join(m_d, "{0}.{1}.fosm.jcb".format(case, pst.control_data.noptmax)))



def plot_zdt_risk_demo_compare(case="zdt1"):
    import matplotlib.pyplot as plt
    # m_d = os.path.join("mou_tests", case + "_test_master_riskobj_more_"+mou_gen)
    truth_func = None
    numdv = 30
    if case == "zdt1":
        truth_func = mou_suite_helper.zdt1
    elif case == "zdt3":
        truth_func = mou_suite_helper.zdt3
    elif case == "zdt6":
        truth_func = mou_suite_helper.zdt6
        numdv = 10
    elif case == "constr":
        numdv = 2
    #else:
    #    raise Exception()

    x0 = np.linspace(-1, 1, 1000)
    ov1, ov2 = [], []
    if truth_func is not None:
        for xx0 in x0:
            x = np.zeros(numdv)
            x[0] = xx0
            ret_vals = truth_func(x)
            ov1.append(ret_vals[0][0])
            ov2.append(ret_vals[0][1])
    master_ds = [case+"_test_master_deter",case+"_test_master_05",case+"_test_master_95"]
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.flatten()
    obj_names = ["obj_1","obj_2"]
    colors = ["g","r","b"]
    labels = ["risk neutral (risk=0.5)","risk tolerant (risk=0.05)","risk averse (risk=0.95)"]
    for m_d,c,label in zip(master_ds,colors,labels):
        m_d = os.path.join("mou_tests",m_d)
        pst = pyemu.Pst(os.path.join(m_d+"_de", case + ".pst"))

        df_de = pd.read_csv(os.path.join(m_d+"_de", case + ".pareto.archive.summary.csv"))
        mxgen = df_de.generation.max()
        print(mxgen)
        df_de = df_de.loc[df_de.generation == mxgen, :]
        df_de = df_de.loc[df_de.nsga2_front == 1, :]

        df_pso = pd.read_csv(os.path.join(m_d + "_pso", case + ".pareto.archive.summary.csv"))
        mxgen = df_pso.generation.max()
        print(mxgen)
        df_pso = df_pso.loc[df_pso.generation == mxgen, :]
        df_pso = df_pso.loc[df_pso.nsga2_front == 1, :]


        axes[0].scatter(df_de.obj_1.values,df_de.obj_2.values,color=c,label=label)
        axes[1].scatter(df_pso.obj_1.values, df_pso.obj_2.values, color=c, label=label)

        if label == labels[0]:


            axes[0].plot(ov1, ov2, "k", label="truth")
            axes[1].plot(ov1, ov2, "k", label="truth")
            #print(x0)
        axes[0].set_title("A) DE specified risk",loc="left")
        axes[1].set_title("A) PSO specified risk", loc="left")
        axes[0].set_xlabel("objective 1")
        axes[1].set_xlabel("objective 1")
        axes[0].set_ylabel("objective 2")
        axes[1].set_ylabel("objective 2")
        axes[1].legend(loc="upper right")

    m_d = os.path.join("mou_tests",case + "_test_master_riskobj_more")
    df_de = pd.read_csv(os.path.join(m_d + "_de", case + ".pareto.archive.summary.csv"))
    mxgen = df_de.generation.max()
    print(mxgen)
    df_de = df_de.loc[df_de.generation == mxgen, :]
    df_de = df_de.loc[df_de.nsga2_front == 1, :]

    df_pso = pd.read_csv(os.path.join(m_d + "_pso", case + ".pareto.archive.summary.csv"))
    mxgen = df_pso.generation.max()
    print(mxgen)
    df_pso = df_pso.loc[df_pso.generation == mxgen, :]
    df_pso = df_pso.loc[df_pso.nsga2_front == 1, :]

    ax = axes[2]
    ax.scatter(df_de.obj_1,df_de.obj_2,marker='.',c=df_de._risk_)
    ax.plot(ov1, ov2, "k", label="truth")
    ax.set_title("C) DE risk objective", loc="left")
    ax.set_ylim(0,10)

    ax = axes[3]
    ax.scatter(df_pso.obj_1, df_pso.obj_2, marker='.', c=df_pso._risk_)
    ax.plot(ov1, ov2, "k", label="truth")
    ax.set_title("D) PSO risk objective", loc="left")
    ax.set_ylim(0,10)
    plt.show()

def zdt1_invest():
    m_d = os.path.join("mou_tests","zdt1_test_master_riskobj_more_pso")
    df_sum = pd.read_csv(os.path.join(m_d,"zdt1.pareto.archive.summary.csv"))
    df_sum = df_sum.loc[df_sum.generation==df_sum.generation.max(),:]
    df_sum = df_sum.loc[df_sum.nsga2_front==1,:]
    df_sum.loc[:,"member"] = df_sum.member.str.lower()
    pst = pyemu.Pst(os.path.join(m_d,"zdt1.pst"))

    pe = pyemu.ParameterEnsemble.from_binary(pst=pst,filename=os.path.join(m_d,"zdt1.archive.dv_pop.jcb"))
    oe = pyemu.ObservationEnsemble.from_binary(pst=pst,filename=os.path.join(m_d,"zdt1.archive.obs_pop.jcb"))
    print(oe.head())
    print(pe.loc[oe.head().index,"_risk_"].head())

    #print(pe.index.shape)
    #print(df_sum.member.shape)

    #print(set(pe.index.tolist()) - set(df_sum.member.tolist()))

    #df_sum.sort_values(by="obj_2",inplace=True,ascending=False)
    #print(df_sum.loc[:,["member","obj_1","obj_2"]].head())
    mem = "gen=144_member=7221_pso"
    pvals = pe.loc[mem,:]
    #print(pvals)
    ovals,cvals = mou_suite_helper.zdt1(pvals.values[:29])
    #print(ovals)
    #print(oe.loc["gen=150_member=7501_pso",:])

    #pst.parameter_data.loc[pvals.index,"parval1"] = pvals.values
    # pst.control_data.noptmax = 1
    # pst.pestpp_options["opt_risk"] = 0.5
    # pst.pestpp_options.pop("mou_risk_objective")
    # pst.pestpp_options["mou_dv_population_file"] = "zdt1.dv_pop.jcb"
    # pst.pestpp_options["mou_obs_population_restart_file"] = "zdt1.obs_pop.jcb"
    # pst.write(os.path.join(m_d,"test.pst"))
    # return
    #pyemu.os_utils.run("pestpp-mou test.pst",cwd=m_d)

    #check that mem is nondominated
    # all objs are minimize
    objs = ["obj_1","obj_2","_risk_"]
    df_sum.loc[:, "_risk_"] *= -1
    mem_vals = df_sum.loc[df_sum.member==mem,objs].values[0]

    print(mem_vals)
    for idx in df_sum.index[1:]:
        mem2 = df_sum.loc[idx,"member"]
        if mem == mem2:
            continue
        mem2_vals = df_sum.loc[idx,objs].values
        d = mem_vals - mem2_vals
        if np.all(d<0):
            print(mem2,mem_vals,mem2_vals)
            #print(d)
        #break


def water_invest():
    case = "water"
    t_d = mou_suite_helper.setup_problem(case, additive_chance=False, risk_obj=False)
    pst = pyemu.Pst(os.path.join(t_d, case + ".pst"))
    pst.pestpp_options["mou_generator"] = "pso"
    pst.pestpp_options["mou_population_size"] = 100
    pst.pestpp_options["opt_risk"] = 0.5
    pst.control_data.noptmax = 200
    pst.write(os.path.join(t_d, case + ".pst"))
    m_d = os.path.join("mou_tests", case + "_pso_master_risk")
    pyemu.os_utils.start_workers(t_d, exe_path, case + ".pst", 50, worker_root="mou_tests",
                                 master_dir=m_d, verbose=True, port=port)

    pst.pestpp_options["mou_generator"] = "de"
    pst.pestpp_options["opt_risk"] = 0.5
    pst.pestpp_options["mou_population_size"] = 100
    pst.control_data.noptmax = 200
    pst.write(os.path.join(t_d, "zdt1.pst"))
    m_d = os.path.join("mou_tests", case + "_de_master_risk")
    pyemu.os_utils.start_workers(t_d, exe_path, case + ".pst", 50, worker_root="mou_tests",
                                 master_dir=m_d, verbose=True, port=port)


def plot_constr_risk():
    import matplotlib.pyplot as plt
    case = "constr"
    master_ds = [case + "_test_master_deter",
                 case + "_test_master_05", case + "_test_master_95"]
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.flatten()
    for i,ax in enumerate(axes):
        label = False
        if i == 1:
            label = True
        get_constr_base_plot(ax,label=label)
        ax.set_xlabel("objective 1 (minimize)")
        ax.set_ylabel("objective 2 (minimize)")
    obj_names = ["obj_1", "obj_2"]
    cmap = plt.get_cmap("viridis")
    colors = [cmap(r) for r in [0.5,0.05,0.95]]
    labels = ["risk neutral (risk=0.5)", "risk tolerant (risk=0.05)", "risk averse (risk=0.95)"]
    for m_d, c, label in zip(master_ds, colors, labels):
        m_d = os.path.join("mou_tests", m_d)
        pst = pyemu.Pst(os.path.join(m_d + "_de", case + ".pst"))
        df_de = pd.read_csv(os.path.join(m_d + "_de", case + ".pareto.archive.summary.csv"))
        mxgen = df_de.generation.max()
        print(mxgen)
        df_de = df_de.loc[df_de.generation == mxgen, :]
        df_de = df_de.loc[df_de.nsga2_front == 1, :]

        df_pso = pd.read_csv(os.path.join(m_d + "_pso", case + ".pareto.archive.summary.csv"))
        mxgen = df_pso.generation.max()
        print(mxgen)
        df_pso = df_pso.loc[df_pso.generation == mxgen, :]
        df_pso = df_pso.loc[df_pso.nsga2_front == 1, :]
        axes[0].scatter(df_de.obj_1.values, df_de.obj_2.values, color=c, s=4, label=label,zorder=10,alpha=0.5)
        axes[1].scatter(df_pso.obj_1.values, df_pso.obj_2.values, color=c, s=4, label=label,zorder=10,alpha=0.5)

        axes[0].set_title("A) DE specified risk", loc="left")
        axes[1].set_title("B) PSO specified risk", loc="left")
        axes[1].legend(loc="upper right",framealpha=1.0)

    m_d = os.path.join("mou_tests", case + "_test_master_riskobj_more")
    df_de = pd.read_csv(os.path.join(m_d + "_de", case + ".pareto.archive.summary.csv"))
    mxgen = df_de.generation.max()
    print(mxgen)
    df_de = df_de.loc[df_de.generation == mxgen, :]
    df_de = df_de.loc[df_de.nsga2_front == 1, :]

    df_pso = pd.read_csv(os.path.join(m_d + "_pso", case + ".pareto.archive.summary.csv"))
    mxgen = df_pso.generation.max()
    print(mxgen)
    df_pso = df_pso.loc[df_pso.generation == mxgen, :]
    df_pso = df_pso.loc[df_pso.nsga2_front == 1, :]

    ax = axes[2]
    ax.scatter(df_de.obj_1, df_de.obj_2, marker='.', c=df_de._risk_,alpha=0.5,s=10)

    ax.set_title("C) DE objective risk", loc="left")
    ax.set_ylim(0, 10)

    ax = axes[3]
    ax.scatter(df_pso.obj_1, df_pso.obj_2, marker='.', c=df_pso._risk_,alpha=0.5,s=10)

    ax.set_title("D) PSO objective risk", loc="left")
    ax.set_ylim(0, 10)
    plt.savefig("constr_results.pdf")

def get_constr_base_plot(ax,label=False,fontsize=10):

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    def f(x1, x2):
        return (1 + x2) / x1

    # first front
    x1 = np.linspace(0.365, 0.665, 100)
    y1 = 6 - 9 * x1
    f1 = f(x1, y1)

    # second front
    x2 = np.linspace(0.1, 1.0, 100)
    f2 = f(x2, 0)

    # top constraint
    y2 = (9 * x1) - 1
    f3 = f(x1, y2)

    # top feas
    x3 = np.linspace(0.67, 1.0, 100)
    f4 = f(x3, 5)

    # right feas
    y4 = np.linspace(0, 5, 100)
    x4 = np.ones_like(y4)
    f5 = f(1, y4)

    # make the polygon

    # fig,ax = plt.subplots(1,1,figsize=(6,6))
    # ax = axes[0]
    if label:
        ax.plot(x1, f1, "k--", label="constraint")
        ax.plot(x2, f2, "r--", label="pareto frontier")
    else:
        ax.plot(x1, f1, "k--")
        ax.plot(x2, f2, "r--")
    ax.plot(x2, f3, "k--")
    # ax.plot(x3,f4,"")
    # ax.plot(x4,f5,"m")
    ax.set_ylim(0, 10)
    ax.set_xlim(0, 1.1)
    # plt.show()

    # get feasible polygon
    x1 = np.linspace(0.39, 0.67, 100)
    y1 = 6 - 9 * x1
    f1 = f(x1, y1)
    x2 = np.linspace(0.39, 0.66, 100)
    x11 = np.linspace(0.0, 1.0, 100)
    y2 = (9 * x1) - 1
    f3 = f(x1, y2)
    x3 = np.linspace(0.665, 1.0, 100)
    f4 = f(x3, 5)
    f2 = f(x3, 0)
    y4 = np.linspace(0, 5, 100)
    x4 = np.ones_like(y4)
    f5 = f(1, y4)
    xvals = list(x3)
    xvals.extend(list(x4))
    xvals.extend(list(np.flipud(x3)))
    xvals.extend(list(np.flipud(x2)))
    xvals.extend(list(x1))
    yvals = list(f2)
    yvals.extend(list(f5))
    yvals.extend(list(np.flipud(f4)))
    yvals.extend(list(np.flipud(f3)))
    yvals.extend(list(f1))
    xy = np.array([xvals, yvals]).transpose()
    if label:
        p = Polygon(xy, facecolor="0.5", alpha=0.5, edgecolor="none",zorder=0, label="feasible region")
    else:
        p = Polygon(xy, facecolor="0.5", alpha=0.5, edgecolor="none", zorder=0)
    ax.add_patch(p)

    ax.set_ylim(0, 10)
    ax.set_xlim(0, 1.1)
    #ax.set_ylabel("objective 1")
    #ax.set_xlabel("objective 2")
    #ax.legend(loc="lower left")


def plot_constr_risk_2():
    import matplotlib.pyplot as plt
    import string
    case = "constr"
    master_ds = [case + "_test_master_riskobj_singlept_more",
                 case + "_test_master_riskobj_allpts_more"]
    labels = ["single chance point","all chance point"]
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    ax_count = 0
    for i,ax in enumerate(axes.flatten()):
        label = False;
        if i == 1:
            label = True
        get_constr_base_plot(ax,label=label)
        ax.set_xlabel("objective 1 (minimize)")
        ax.set_ylabel("objective 2 (minimize)")
    for irow,(m_d,lab) in enumerate(zip(master_ds,labels)):
        m_d = os.path.join("mou_tests", m_d)
        pst = pyemu.Pst(os.path.join(m_d + "_de", case + ".pst"))
        df_de = pd.read_csv(os.path.join(m_d + "_de", case + ".pareto.archive.summary.csv"))
        mxgen = df_de.generation.max()
        print(mxgen)
        df_de = df_de.loc[df_de.generation == mxgen, :]
        df_de = df_de.loc[df_de.nsga2_front == 1, :]

        df_pso = pd.read_csv(os.path.join(m_d + "_pso", case + ".pareto.archive.summary.csv"))
        mxgen = df_pso.generation.max()
        print(mxgen)
        df_pso = df_pso.loc[df_pso.generation == mxgen, :]
        df_pso = df_pso.loc[df_pso.nsga2_front == 1, :]
        axes[irow,0].scatter(df_de.obj_1.values, df_de.obj_2.values, c=df_de._risk_, s=4, label=label,zorder=10,alpha=0.5)
        axes[irow,1].scatter(df_pso.obj_1.values, df_pso.obj_2.values, c=df_pso._risk_, s=4, label=label,zorder=10,alpha=0.5)

        axes[irow,0].set_title("{0}) DE {1}".format(string.ascii_uppercase[ax_count],lab), loc="left")
        ax_count += 1
        axes[irow,1].set_title("{0}) PSO {1}".format(string.ascii_uppercase[ax_count],lab), loc="left")
        axes[irow,1].legend(loc="upper right",framealpha=1.0)



    plt.savefig("constr_results_2.pdf")

def plot_constr_risk_3():
    import matplotlib.pyplot as plt
    import string
    case = "constr"
    master_ds = [case + "_test_master_allpts_every_riskobj_more",
                 case + "_test_master_riskobj_allpts_more"]
    labels = ["all chance point, every", "all chance point, reuse"]
    fig, axes = plt.subplots(4, 2, figsize=(8, 11))
    ax_count = 0
    for i,ax in enumerate(axes.flatten()):
        label = False;
        if i == 1:
            label = True
        get_constr_base_plot(ax,label=label)
        ax.set_xlabel("objective 1 (minimize)")
        ax.set_ylabel("objective 2 (minimize)")
    for irow,(m_d,lab) in enumerate(zip(master_ds,labels)):
        m_d = os.path.join("mou_tests", m_d)
        pst = pyemu.Pst(os.path.join(m_d + "_de", case + ".pst"))
        df_de = pd.read_csv(os.path.join(m_d + "_de", case + ".pareto.archive.summary.csv"))
        mxgen = df_de.generation.max()
        print(mxgen)
        df_de = df_de.loc[df_de.generation == mxgen, :]
        df_de = df_de.loc[df_de.nsga2_front == 1, :]

        df_pso = pd.read_csv(os.path.join(m_d + "_pso", case + ".pareto.archive.summary.csv"))
        mxgen = df_pso.generation.max()
        print(mxgen)
        df_pso = df_pso.loc[df_pso.generation == mxgen, :]
        df_pso = df_pso.loc[df_pso.nsga2_front == 1, :]
        axes[irow,0].scatter(df_de.obj_1.values, df_de.obj_2.values, c=df_de._risk_, s=4, label=label,zorder=10,alpha=0.5)
        axes[irow,1].scatter(df_pso.obj_1.values, df_pso.obj_2.values, c=df_pso._risk_, s=4, label=label,zorder=10,alpha=0.5)

        axes[irow,0].set_title("{0}) DE {1}".format(string.ascii_uppercase[ax_count],lab), loc="left")
        ax_count += 1
        axes[irow,1].set_title("{0}) PSO {1}".format(string.ascii_uppercase[ax_count],lab), loc="left")
        axes[irow,1].legend(loc="upper right",framealpha=1.0)


    plt.tight_layout()
    plt.savefig("constr_results_3.pdf")



def plot_constr_risk_pub():
    import string
    import matplotlib.pyplot as plt
    case = "constr"
    fs = 9
    master_ds = [case + "_test_master_deter",
                 case + "_test_master_05", case + "_test_master_95"]
    fig, axes = plt.subplots(4, 2, figsize=(8,11))
    for i,ax in enumerate(axes.flatten()):
        label = False
        if i == 1:
            label = True
        get_constr_base_plot(ax,label=label,fontsize=fs)
        ax.set_xlabel("$f_1$ (minimize)",fontsize=fs)
        ax.set_ylabel("$f_2$ (minimize)",fontsize=fs)
    obj_names = ["obj_1", "obj_2"]
    cmap = plt.get_cmap("viridis")
    colors = [cmap(r) for r in [0.5,0.05,0.95]]
    labels = ["risk neutral\n(risk=0.5)", "risk tolerant\n(risk=0.05)", "risk averse\n(risk=0.95)"]
    
    for m_d, c, label in zip(master_ds, colors, labels):
        m_d = os.path.join("mou_tests", m_d)
        pst = pyemu.Pst(os.path.join(m_d + "_de", case + ".pst"))
        df_de = pd.read_csv(os.path.join(m_d + "_de", case + ".pareto.archive.summary.csv"))
        mxgen = df_de.generation.max()
        print(mxgen)
        df_de = df_de.loc[df_de.generation == mxgen, :]
        df_de = df_de.loc[df_de.nsga2_front == 1, :]

        df_pso = pd.read_csv(os.path.join(m_d + "_pso", case + ".pareto.archive.summary.csv"))
        mxgen = df_pso.generation.max()
        print(mxgen)
        df_pso = df_pso.loc[df_pso.generation == mxgen, :]
        df_pso = df_pso.loc[df_pso.nsga2_front == 1, :]
        axes[0,0].scatter(df_de.obj_1.values, df_de.obj_2.values, color=c, s=4, label=label,zorder=10,alpha=0.5)
        axes[0,1].scatter(df_pso.obj_1.values, df_pso.obj_2.values, color=c, s=4, label=label,zorder=10,alpha=0.5)

        axes[0,0].set_title("A) DE specified risk, specified uncertainty", loc="left",fontsize=fs)
        axes[0,1].set_title("B) PSO specified risk, specified uncertainty", loc="left",fontsize=fs)
        axes[0,1].legend(loc="upper right",framealpha=1.0,fontsize=8)

    master_ds = [case+"_test_master_riskobj_more",
                 case + "_test_master_riskobj_singlept_more",
                 case + "_test_master_riskobj_allpts_more",
                 ]
    labels = ["objective risk, specified uncertainty",
              "objective risk, stack-based uncertainty,\n     single chance point, reused across generations",
              "objective risk, stack-based uncertainty,\n     all chance point, reused across generations"]
    ax_count = 2
    for irow,(m_d,lab) in enumerate(zip(master_ds,labels)):
        irow += 1
        m_d = os.path.join("mou_tests", m_d)
        pst = pyemu.Pst(os.path.join(m_d + "_de", case + ".pst"))
        df_de = pd.read_csv(os.path.join(m_d + "_de", case + ".pareto.archive.summary.csv"))
        mxgen = df_de.generation.max()
        print(mxgen)
        df_de = df_de.loc[df_de.generation == mxgen, :]
        df_de = df_de.loc[df_de.nsga2_front == 1, :]

        df_pso = pd.read_csv(os.path.join(m_d + "_pso", case + ".pareto.archive.summary.csv"))
        mxgen = df_pso.generation.max()
        print(mxgen)
        df_pso = df_pso.loc[df_pso.generation == mxgen, :]
        df_pso = df_pso.loc[df_pso.nsga2_front == 1, :]
        axes[irow,0].scatter(df_de.obj_1.values, df_de.obj_2.values, c=df_de._risk_, s=4, label=label,zorder=10,alpha=0.5)
        axes[irow,1].scatter(df_pso.obj_1.values, df_pso.obj_2.values, c=df_pso._risk_, s=4, label=label,zorder=10,alpha=0.5)

        axes[irow,0].set_title("{0}) DE {1}".format(string.ascii_uppercase[ax_count],lab), loc="left",fontsize=fs)
        ax_count += 1
        axes[irow,1].set_title("{0}) PSO {1}".format(string.ascii_uppercase[ax_count],lab), loc="left",fontsize=fs)
        ax_count += 1
        #axes[irow,1].legend(loc="upper right",framealpha=1.0)


    plt.tight_layout()
    plt.savefig("constr_results_pub.pdf")


def pop_sched_test():
    case = "zdt1"
    t_d = mou_suite_helper.setup_problem(case, additive_chance=True, risk_obj=False)
    pst = pyemu.Pst(os.path.join(t_d, case + ".pst"))
    pst.control_data.noptmax = 3
    pst.write(os.path.join(t_d, case + ".pst"))
    m_d = os.path.join("mou_tests",case+"_master_pop_sched")
    pyemu.os_utils.start_workers(t_d, exe_path, case + ".pst", 50, worker_root="mou_tests",
                                 master_dir=m_d, verbose=True, port=port)


def simplex_invest_1():
    case = "zdt1"
    t_d = mou_suite_helper.setup_problem(case, additive_chance=False, risk_obj=False)
    pst = pyemu.Pst(os.path.join(t_d, case + ".pst"))
    pst.pestpp_options["mou_generator"] = "de,simplex"
    pst.pestpp_options["mou_simplex_reflections"] = 10
    pst.control_data.noptmax = 100

    pst.write(os.path.join(t_d, case + ".pst"))
    m_d = os.path.join("mou_tests",case+"_master_simplex1")

    pyemu.os_utils.start_workers(t_d, exe_path, case + ".pst", 50, worker_root="mou_tests",
                                 master_dir=m_d, verbose=True, port=port)



if __name__ == "__main__":
        
    #shutil.copy2(os.path.join("..","exe","windows","x64","Debug","pestpp-mou.exe"),os.path.join("..","bin","pestpp-mou.exe"))
    #basic_pso_test()

    #shutil.copy2(os.path.join("..", "bin", "win", "pestpp-mou.exe"),
    #             os.path.join("..", "bin", "pestpp-mou.exe"))
    #basic_pso_test()
    #risk_obj_test()
    #invest_2()
    #chance_consistency_test()
    #invest_3()
    # mou_suite_helper.start_workers("zdt1")
    #all_infeas_test()
    #invest_4()
    #restart_dv_test()
    #chance_all_binary_test()
    #invest_5()
    #constr_risk_demo()
    #plot_constr_risk_demo()

    #risk_demo(case='kur',noptmax=300,std_weight=0.01)
    #plot_risk_demo_multi(case='kur')

    #risk_demo(case='zdt1',noptmax=300,std_weight=0.00001)
    #plot_risk_demo_multi(case='zdt1')

    #risk_demo(case="rosenc",std_weight=1.0,noptmax=500)
    #plot_risk_demo_multi()
    #plot_risk_demo_rosen()
    #risk_demo(case='constr', noptmax=150, std_weight=0.05, pop_size=100, num_workers=50, mou_gen="de")
    #risk_demo(case='constr', noptmax=150, std_weight=0.05, pop_size=100, mou_gen="de", num_workers=50)

    #risk_demo(case='constr',noptmax=20,std_weight=0.05,pop_size=100,num_workers=50,mou_gen="simplex,de")
    #risk_demo(case='constr', noptmax=20, std_weight=0.05, pop_size=100,mou_gen="de",num_workers=50)
    #plot_risk_demo_multi_3pane(case='zdt1',mou_gen="de")
    #plot_zdt_risk_demo_compare(case="constr")
    #zdt1_invest()
    #test_sorting_fake_problem()
    #plot_risk_demo_rosen()
    #risk_demo(case="rosenc",std_weight=1.0,mou_gen="pso",pop_size=100,noptmax=30)
    #plot_risk_demo_rosen(mou_gen="pso")
    #risk_demo(case="rosenc", std_weight=1.0, mou_gen="de",pop_size=100,noptmax=30)
    #plot_risk_demo_rosen(mou_gen="de")
    #all_infeas_test()
    #case = "constr"
    #basic_pso_test(case=case)
    #water_invest()
    #mou_suite_helper.plot_results(os.path.join("mou_tests",case+"_pso_master_risk"),sequence=True)
    #mou_suite_helper.plot_results(os.path.join("mou_tests",case+"_de_master_risk"),sequence=True)
    #plot_constr_risk()
    plot_constr_risk_pub()
    #stack_map_invest()

    #pop_sched_test()
    #simplex_invest_1()