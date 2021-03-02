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
    pyemu.os_utils.run("{0} tkn.pst".format(exe_path),cwd=t_d)
    out_file = os.path.join(t_d,"tkn.obs_pop.csv".format(pst.control_data.noptmax))
    assert os.path.exists(out_file)
    df = pd.read_csv(out_file)
    assert df.shape[0] == pst.pestpp_options["mou_population_size"]

    pst.pestpp_options["mou_generator"] = "sbx"
    pst.pestpp_options["mou_env_selector"] = "spea"
    pst.write(os.path.join(t_d,"tkn.pst"))
    pyemu.os_utils.run("{0} tkn.pst".format(exe_path),cwd=t_d)
    out_file = os.path.join(t_d,"tkn.obs_pop.csv".format(pst.control_data.noptmax))
    assert os.path.exists(out_file)
    df = pd.read_csv(out_file)
    assert df.shape[0] == pst.pestpp_options["mou_population_size"]

    pst.pestpp_options["mou_generator"] = "de"
    pst.pestpp_options["mou_env_selector"] = "nsga"
    pst.write(os.path.join(t_d,"tkn.pst"))
    pyemu.os_utils.run("{0} tkn.pst".format(exe_path),cwd=t_d)
    out_file = os.path.join(t_d,"tkn.obs_pop.csv".format(pst.control_data.noptmax))
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



def risk_demo(case="zdt1",noptmax=100,std_weight=0.0001):

    obj_names = ["obj_1"]
    if "zdt" in case:
        obj_names.append("obj_2")
    t_d = mou_suite_helper.setup_problem(case, additive_chance=False, risk_obj=False)
    pst = pyemu.Pst(os.path.join(t_d, case+".pst"))
    #pst.pestpp_options["opt_chance_points"] = "all"
    pst.pestpp_options["opt_recalc_chance_every"] = 100000
    #pst.pestpp_options["opt_stack_size"] = 100
    pst.pestpp_options["mou_generator"] = "de"
    pst.pestpp_options["mou_population_size"] = 100
    pst.pestpp_options["opt_risk"] = 0.5
    pst.pestpp_options["save_binary"] = True
    pst.observation_data.loc[pst.nnz_obs_names,"weight"] = std_weight
    pst.pestpp_options["opt_std_weights"] = True
    pst.control_data.noptmax = noptmax
    pst.write(os.path.join(t_d, case+".pst"))
    m1 = os.path.join("mou_tests", case+"_test_master_deter")
    pyemu.os_utils.start_workers(t_d, exe_path, case+".pst", 50, worker_root="mou_tests",
                                 master_dir=m1, verbose=True, port=port)

    t_d = mou_suite_helper.setup_problem(case, additive_chance=True, risk_obj=False)
    pst = pyemu.Pst(os.path.join(t_d, case+".pst"))
    #pst.pestpp_options["opt_chance_points"] = "all"
    #pst.pestpp_options["opt_recalc_chance_every"] = 100000
    #pst.pestpp_options["opt_stack_size"] = 100
    pst.observation_data.loc[pst.nnz_obs_names,"weight"] = std_weight
    pst.pestpp_options["opt_std_weights"] = True
    pst.pestpp_options["mou_generator"] = "de"
    pst.pestpp_options["mou_population_size"] = 100
    pst.pestpp_options["opt_risk"] = 0.95
    pst.pestpp_options["save_binary"] = True
    pst.control_data.noptmax = noptmax
    pst.write(os.path.join(t_d, case+".pst"))
    m2 = os.path.join("mou_tests", case+"_test_master_95")
    pyemu.os_utils.start_workers(t_d, exe_path, case+".pst", 50, worker_root="mou_tests",
                                 master_dir=m2, verbose=True, port=port)

    t_d = mou_suite_helper.setup_problem(case, additive_chance=True, risk_obj=False)
    pst = pyemu.Pst(os.path.join(t_d, case+".pst"))
    #pst.pestpp_options["opt_chance_points"] = "all"
    #pst.pestpp_options["opt_recalc_chance_every"] = 100000
    #pst.pestpp_options["opt_stack_size"] = 100
    pst.observation_data.loc[pst.nnz_obs_names,"weight"] = std_weight
    pst.pestpp_options["opt_std_weights"] = True
    pst.pestpp_options["mou_generator"] = "de"
    pst.pestpp_options["mou_population_size"] = 100
    pst.pestpp_options["opt_risk"] = 0.05
    pst.pestpp_options["save_binary"] = True
    pst.control_data.noptmax = noptmax
    pst.write(os.path.join(t_d, case+".pst"))
    m3 = os.path.join("mou_tests", case+"_test_master_05")
    pyemu.os_utils.start_workers(t_d, exe_path, case+".pst", 50, worker_root="mou_tests",
                                 master_dir=m3, verbose=True, port=port)

    t_d = mou_suite_helper.setup_problem(case, additive_chance=True, risk_obj=True)
    pst = pyemu.Pst(os.path.join(t_d, case+".pst"))
    #pst.pestpp_options["opt_chance_points"] = "all"
    #pst.pestpp_options["opt_recalc_chance_every"] = 100000
    #pst.pestpp_options["opt_stack_size"] = 100
    pst.observation_data.loc[pst.nnz_obs_names,"weight"] = std_weight
    pst.pestpp_options["opt_std_weights"] = True
    pst.pestpp_options["mou_generator"] = "de"
    pst.pestpp_options["mou_population_size"] = 100
    pst.pestpp_options["opt_risk"] = 0.05
    pst.pestpp_options["save_binary"] = True
    pst.control_data.noptmax = noptmax
    pst.write(os.path.join(t_d, case+".pst"))
    m4 = os.path.join("mou_tests", case+"_test_master_riskobj_match")
    pyemu.os_utils.start_workers(t_d, exe_path, case+".pst", 50, worker_root="mou_tests",
                                 master_dir=m4, verbose=True, port=port)

    t_d = mou_suite_helper.setup_problem(case, additive_chance=True, risk_obj=True)
    pst = pyemu.Pst(os.path.join(t_d, case+".pst"))
    #pst.pestpp_options["opt_chance_points"] = "all"
    #pst.pestpp_options["opt_recalc_chance_every"] = 100000
    #pst.pestpp_options["opt_stack_size"] = 100
    pst.observation_data.loc[pst.nnz_obs_names,"weight"] = std_weight
    pst.pestpp_options["opt_std_weights"] = True
    pst.pestpp_options["mou_generator"] = "de"
    pst.pestpp_options["mou_population_size"] = 100
    pst.pestpp_options["opt_risk"] = 0.05
    pst.pestpp_options["save_binary"] = True
    pst.control_data.noptmax = noptmax * 3
    pst.write(os.path.join(t_d, case+".pst"))
    m5 = os.path.join("mou_tests", case+"_test_master_riskobj_more")
    pyemu.os_utils.start_workers(t_d, exe_path, case+".pst", 50, worker_root="mou_tests",
                                 master_dir=m5, verbose=True, port=port)

def plot_risk_demo_multi(case = "zdt1"):
    import matplotlib.pyplot as plt
    m_deter = os.path.join("mou_tests",case+"_test_master_deter")
    m_ravr = os.path.join("mou_tests",case+"_test_master_95")
    m_rtol = os.path.join("mou_tests", case+"_test_master_05")
    m_robj = os.path.join("mou_tests",case+"_test_master_riskobj_match")
    m_robjm = os.path.join("mou_tests", case+"_test_master_riskobj_more")

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


def plot_risk_demo_multi_3pane(case="zdt1"):
    import matplotlib.pyplot as plt
    m_d = os.path.join("mou_tests", case + "_test_master_riskobj_more")
    pst = pyemu.Pst(os.path.join(m_d, case + ".pst"))
    obj_names = pst.pestpp_options["mou_objectives"].lower().split(',')
    df = pd.read_csv(os.path.join(m_d, case + ".pareto.archive.summary.csv"))
    mxgen = df.generation.max()
    print(mxgen)
    df = df.loc[df.generation == mxgen, :]
    df = df.loc[df.nsga2_front==1,:]
    
    #df = pyemu.ObservationEnsemble.from_binary(pst=pst,filename=os.path.join(m_d,case+".archive.obs_pop.jcb"))
    #pe = pyemu.ParameterEnsemble.from_binary(pst=pst,filename=os.path.join(m_d,case+".archive.dv_pop.jcb"))
    #print(pe.loc[:,"dv_1"].min())
    #return
    #df.loc[:,"_risk_"] = pe.loc[df.index,"_risk_"].values
    #df = df.loc[df._risk_.apply(lambda x: x > 0.95),:]
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
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
                ax.scatter(v2,v1,marker=".",color="0.5",s=20)
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

            ax.scatter(v2,v1,marker=".",color="0.5",s=20)
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
    m_deter = os.path.join("mou_tests",case+"_test_master_deter")
    m_ravr = os.path.join("mou_tests",case+"_test_master_95")
    m_rtol = os.path.join("mou_tests", case+"_test_master_05")
    #m_robj = os.path.join("mou_tests",case+"_test_master_riskobj_match")
    m_robjm = os.path.join("mou_tests", case+"_test_master_riskobj_match")
    bins = np.linspace(-5,5,30)
    fig, axes = plt.subplots(2,2,figsize=(10,10))
    axes = axes.flatten()
    for d,c,ax in zip([m_deter,m_ravr,m_rtol,m_robjm],['g','b','r',"c","m"],axes):

        pst = pyemu.Pst(os.path.join(d,case+".pst"))
        df = pd.read_csv(os.path.join(d,case+".pareto.archive.summary.csv"))
        mxgen = df.generation.max()
        #mxgen = 10
        #print(d,mxgen)
        df = df.loc[df.generation==mxgen,:]
        print(d,mxgen)
        if "riskobj" in d:
            #print(df.head().loc[:,['obj_1',"obj_2",'_risk_']])
            #ax.scatter(df.obj_1.values[:2],df.obj_2.values[:2],marker="+",c=df._risk_[:2],cmap='jet')
            rdf = df#.loc[df._risk_ < 0.05,:]
            axt = plt.twinx(ax)
            axt.scatter(rdf.obj_1,rdf._risk_,marker="o",c=1 - rdf._risk_.values,cmap='jet')

        ax.hist(df.obj_1.values,bins=bins,facecolor=c,edgecolor="none",alpha=0.5,density=False)
        ax.set_xlim(bins.min(),bins.max())
        ax.set_title(d)
    plt.show()

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


if __name__ == "__main__":
        
    shutil.copy2(os.path.join("..","exe","windows","x64","Debug","pestpp-mou.exe"),os.path.join("..","bin","pestpp-mou.exe"))
    #shutil.copy2(os.path.join("..", "bin", "win", "pestpp-mou.exe"),
    #             os.path.join("..", "bin", "pestpp-mou.exe"))

    risk_obj_test()
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

    risk_demo(case='kur',noptmax=300,std_weight=0.01)
    plot_risk_demo_multi(case='kur')

    #risk_demo(case='zdt1',noptmax=300,std_weight=0.00001)
    #plot_risk_demo_multi(case='zdt1')


    #risk_demo(case="rosenc",std_weight=1.0,noptmax=500)
    #plot_risk_demo_multi()
    #plot_risk_demo_rosen()
    #risk_demo(case='zdt1',noptmax=300,std_weight=0.00001)
    #plot_risk_demo_multi_3pane(case='zdt1')

    #plot_risk_demo_rosen()
    #risk_demo(case="rosenc",std_weight=1.0)
    #plot_risk_demo_multi()
    #plot_risk_demo_rosen(case="rosenc")
