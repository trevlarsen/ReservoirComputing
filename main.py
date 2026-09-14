import sys
import os
sys.path.insert(0, os.path.abspath(f'{os.getcwd()}/utils/'))
import utils.helper as helper
import utils.driver as driver

"""
Main Method to call the Gridsearch
"""

# TODO: Explore Giant Component - create new set with erdos c being 2 so giant component is half before and the half after is the split and include 100% thinning
# TODO: Line plot the first column (rhos on the x-axis) metrics like VPT, consistency, and diversities on the y-axis
# TODO: Top right corner - higher highs and lower lows?

def main():
    network_type, rho_p_thin_set, param, param_name, param_set = helper.parse_arguments()

    rho_p_thin_prod, erdos_possible_combinations = helper.generate_params(
        rho_p_thin_set,
        param=param, 
        param_name=param_name,
        param_set=param_set
    )

    n, _ = rho_p_thin_prod.shape

    if n == 1:
        rho, p_thin = rho_p_thin_prod[0]
    else:
        job_id_number = int(os.getenv('ID_TO_PROCESS'))
        print(job_id_number)
        rho, p_thin = rho_p_thin_prod[job_id_number]
        
    home = os.path.expanduser("~")
    results_path = f'{home}/nobackup/autodelete/results/{network_type}/{param_name}/{param}/{param_set}/{rho_p_thin_set}/'

    # Production default: run draws for up to 2 hours per (rho, p_thin).
    # Test submits can override with --export=TF_SECONDS=300
    tf = float(os.getenv("TF_SECONDS", "7200"))
    draw_count = int(os.getenv("DRAW_COUNT", "100000"))
    print(f"tf={tf} draw_count={draw_count} results_path={results_path}")

    driver.rescomp_parallel_uniform_gridsearch_h5(
        network_type,
        erdos_possible_combinations, 
        rho,
        p_thin,
        draw_count=draw_count, 
        hdf5_file_path=results_path, 
        tf=tf
    )


if __name__ == "__main__":
    main()