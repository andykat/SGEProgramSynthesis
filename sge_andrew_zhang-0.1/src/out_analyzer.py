import os

successes = 0
sum_iterations = 0
tries = 0
pool_count = os.cpu_count()

for file_index in range(pool_count+5):
    file_name = "out" + str(file_index) + ".csv"
    try:
        with open(file_name) as fp:
            i = 0
            for line in fp:
                tries += 1
                print(i)
                i += 1

                sep = line.split(',')
                success = sep[0]
                iteration = int(sep[1])
                print(line)
                if success == "True":
                    print("success")
                    successes += 1
                    sum_iterations += iteration
                print(successes)
    except FileNotFoundError:
        pass

t_str = "Successes: " + str(successes)
print(t_str)
t_str = "Tries: " + str(tries)
print(t_str)
avg_iterations = 0
if sum_iterations > 0:
    avg_iterations = float(sum_iterations) / float(successes)
t_str = "avg iterations: " + str(avg_iterations)
print(t_str)