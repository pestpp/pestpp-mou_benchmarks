import os
import sys
import shutil
import platform
import numpy as np
import pandas as pd
import platform
import pyemu

import mou_suite_helper

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
                                 master_dir=m1,verbose=True)

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
                                 verbose=True)

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
                                 master_dir=m1, verbose=True)

    shutil.copy2(os.path.join(m1,'zdt1.0.par_stack.csv'),os.path.join(t_d,"par_stack.csv"))
    shutil.copy2(os.path.join(m1, 'zdt1.0.obs_stack.csv'), os.path.join(t_d, "obs_stack.csv"))


    pst.pestpp_options["opt_par_stack"] = "par_stack.csv"
    pst.pestpp_options["opt_obs_stack"] = "obs_stack.csv"
    pst.control_data.noptmax = 3
    pst.pestpp_options["opt_recalc_chance_every"] = 2
    pst.write(os.path.join(t_d, "zdt1.pst"))
    m2 = os.path.join("mou_tests", "zdt1_test_master_restart2")
    pyemu.os_utils.start_workers(t_d, exe_path, "zdt1.pst", 35, worker_root="mou_tests",
                                 master_dir=m2, verbose=True)

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
                                 master_dir=m1, verbose=True)

    shutil.copy2(os.path.join(m1, 'zdt1.0.nested.par_stack.csv'), os.path.join(t_d, "par_stack.csv"))
    shutil.copy2(os.path.join(m1, 'zdt1.0.nested.obs_stack.csv'), os.path.join(t_d, "obs_stack.csv"))

    pst.pestpp_options["opt_par_stack"] = "par_stack.csv"
    pst.pestpp_options["opt_obs_stack"] = "obs_stack.csv"
    pst.control_data.noptmax = 3
    pst.pestpp_options["opt_recalc_chance_every"] = 2
    pst.write(os.path.join(t_d, "zdt1.pst"))
    m2 = os.path.join("mou_tests", "zdt1_test_master_restart2")
    pyemu.os_utils.start_workers(t_d, exe_path, "zdt1.pst", 35, worker_root="mou_tests",
                                 master_dir=m2, verbose=True)

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
                                 master_dir=m1,verbose=True)
    plot_results(os.path.join("mou_tests", "zdt1_test_master_riskobj_full"))


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
                                 master_dir=m1, verbose=True)

    t_d = setup_problem("zdt1", additive_chance=True, risk_obj=False)
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
                                 master_dir=m1, verbose=True)




if __name__ == "__main__":
        
    #zdt1_test()
    # setup_zdt_problem("zdt1",30)
    # setup_zdt_problem("zdt2",30)
    # setup_zdt_problem("zdt3",30)
    # setup_zdt_problem("zdt4",10)
    # setup_zdt_problem("zdt6",10)
    shutil.copy2(os.path.join("..","exe","windows","x64","Debug","pestpp-mou.exe"),os.path.join("..","bin","pestpp-mou.exe"))

    #shutil.copy2(os.path.join("..","bin","win","pestpp-mou.exe"),os.path.join("..","bin","pestpp-mou.exe"))
    
    #for case in ["srn","constr","zdt4","zdt3","zdt2","zdt1"]:
    #   master_d = run_problem(case,noptmax=100)
    #   plot_results(master_d)

    #setup_problem("srn",additive_chance=True)
    #master_d = run_problem_chance("srn",noptmax=5,chance_points="all",pop_size=10,stack_size=10,recalc=3)
    #plot_results(os.path.join("mou_tests","zdt1_invest"))
    #plot_results(os.path.join("mou_tests", "zdt1_test_master_riskobj_full"),sequence=True)
    #invest_risk_obj()
    #master_d = os.path.join("mou_tests","zdt6_master")
    #plot_results(master_d)
    #for case in ["zdt1","zdt2","zdt3","zdt4","zdt6","sch","srn","constr"]:
    #  master_d = run_problem_chance(case,noptmax=100)
    #  plot_results(master_d)

    #test_setup_and_three_iters()
    #test_restart_single()
    #test_restart_all()
    #setup_problem("water",additive_chance=True, risk_obj=True)
    #setup_problem("zdt1",30, additive_chance=True)
    #test_sorting_fake_problem()
    #start_workers()
    #setup_problem("zdt1")
    #run_problem_chance_external_fixed("zdt1")


    fail_test()
    #run_problem_chance()
    #invest_risk_obj()
    #plot_results(os.path.join("mou_tests","zdt1_test_master"))
    #plot_results(os.path.join("mou_tests", "zdt1_test_master_riskobj"))
    #invest()
    # run_problem_chance("zdt1",chance_points="all", noptmax=300,
    #                    recalc=1000,risk_obj=True)
    #
    # for r in np.linspace(0.011,0.985,20):
    #     if r == 0.5:
    #         r = 0.51
    #     run_problem_chance("zdt1", chance_points="all", noptmax=100,
    #                        recalc=1000, risk_obj=False,risk=r)
    # plot_results(os.path.join("mou_tests","zdt1_master"))
    #risk_compare_plot()
    #plot_results(os.path.join("mou_tests","zdt1_master_chance"))
    
    #run_problem_chance("constr",noptmax=10,risk_obj=True)
    #plot_results(os.path.join("mou_tests","zdt1_master"))
    #setup_problem("constr")
    #run_problem("constr",noptmax=100)
    #master_d = run_single_obj_sch_prob(risk_obj=True)
    #master_d = os.path.join("mou_tests","sch_master")
    #plot_results_single(master_d)
    #setup_problem("ackley")
    #run_problem("ackley")
    #master_d = os.path.join("mou_tests","rosen_master")
    #plot_results_single(master_d)

