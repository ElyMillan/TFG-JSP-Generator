export interface FormData {
    email: string;
    full_name: string;
    school_name: string;

    size: string;
    jobs: string;
    job_step: string;
    machines: string;
    machine_step: string;
    distributions: string[];
    speed_scaling: string;
    release_due_date: string;
    seeds: string;

    pickle_file_output: boolean;
    json_file_output: boolean;
    dzn_file_output: boolean;
    taillard_file_output: boolean;
    single_folder_output: boolean;
    custom_folder_name: string;
    solver: string
}
