# SGE_PROGSYS
structured GE for program synthesis

run src/SGE_Main.py to run sge on a progsys problem.
Variable definitions are set in SGE_Main.py and SGE.py
sge_iterations is the maximum number of iterations for sge to reach the desired
fitness
test_iterations is how many times to let sge solve the problem
It will run scripts on each processor. Make sure the test_iterations is a
 multiple of the number of processors

the results of each test are output to out.csv
run src/outanalyzer.py to see how many times sge solved the problem.

Notable Changes from PonyGE/Progsys
- Renamed the embedded code to problemName_Helper.txt
- Added a <train> variable to the Helper codes to insert train and test datasets
- In the evolve function in the Helper codes, the definitions of local variables
are done in the evolve function, and removed from the first line of each grammar
- Replaced <, >, <=, >= with text characters 'greater_than', 'less_than', etc. 
to prevent confusion from '<' and '>' used in defining nonterminals
- Added a <blank> nonterminal in case of recursive phenotype translation incompletion
