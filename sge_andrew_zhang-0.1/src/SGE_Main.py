from multiprocessing import Pool
import os
import sys
# sys.setrecursionlimit(1500)


def run_test_iterations(input_var):
    success_save = []
    iterations_save = []
    sge = SGE.SGE(input_var["problem_name"])
    sge.population_size = input_var["population_size"]
    sge.recursion_max = input_var["recursion_max"]
    sge.read_bnf_file()
    sge.read_supplementary_files()

    for t in range(0, input_var["test_iterations"]):
        print("test %d" % t)
        results = sge.run_iterations(input_var["sge_iterations"])
        success_save.append(results[2])
        iterations_save.append(results[3])
        # print(results[4])
        print("Best Fitness:")
        print(results[1][0])
        print("\n")
        print("Best Code:")
        print(results[0][0])

        # print("\n\n\n\n\n\n\n\n\nPrinting Best Results\n\n")
        # for i in range(0, 5):
        #     phen = sge.translateSeqToPhenotype(results[0][i])[0][0]
        #     code = sge.translateObjectsIntoCode(phen)
        #     print(code)
        #     print(str(bestResults[1][i]))
        #     print("\n\n")
    f_name = "out" + str(input_var["index"]) + ".csv"
    with open(f_name, "w") as fp:
        for t in range(0, input_var["test_iterations"]):
            t_str = str(success_save[t]) + "," + str(iterations_save[t]) + "\n"
            fp.write(t_str)


if __name__ == '__main__':
    import SGE

    problem_name = "NumberIO"
    population_size = 100  # size of population
    recursion_max = 10  # level of recursion
    sge_iterations = 20  # number of generations
    test_iterations = 4  # number of repeated tests

    pool_count = os.cpu_count()
    pool = Pool(pool_count)
    pool_inputs = []
    for i in range(0, pool_count):
        pool_inputs.append({"index": i, "sge_iterations": sge_iterations, "problem_name": problem_name,
                            "test_iterations": int(test_iterations/pool_count),
                            "population_size": population_size, "recursion_max": recursion_max})

    pool.map(run_test_iterations, pool_inputs)

    # here we only run 1 test on one thread

    # run_test_iterations({"index": 0, "sge_iterations": sge_iterations, "problem_name": problem_name,
    #                      "test_iterations": test_iterations,
    #                      "population_size": population_size, "recursion_max": recursion_max})
