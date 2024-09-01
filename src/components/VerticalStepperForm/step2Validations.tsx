import * as Yup from 'yup';

export const validationStep2 = Yup.object({
    size: Yup.string()
        .matches(/^\d{1,4}$/, "Only numbers are allowed and maximum 5 digits")
        .required('Size field is Required'),

    seeds: Yup.string()
        .matches(/^\d{1,20}$/, "Only numbers are allowed and maximum 20 digits")
        .required('Seed field is Required'),

    jobs: Yup.string()
        .matches(/^\d{1,4}$/, "Only numbers are allowed and maximum 5 digits")
        .required('Jobs field is Required'),

    job_step: Yup.string()
        .matches(/^\d{1,3}$/, "Only numbers are allowed and maximum 5 digits")
        .required('Job Step field is Required'),

    machines: Yup.string()
        .matches(/^\d{1,4}$/, "Only numbers are allowed and maximum 5 digits")
        .required('Machines field is Required'),

    machine_step: Yup.string()
        .matches(/^\d{1,3}$/, "Only numbers are allowed and maximum 5 digits")
        .required('Machine Step field is Required'),

    distributions: Yup.array()
        .min(1, "At least one distribution must be selected")
        .required('Distributions are Required'),

    speed_scaling: Yup.string()
        .matches(/^(?:[1-5])$/, "Must be an integer between 1 and 5")
        .required('Speed Scaling field is Required'),
    
    release_due_date: Yup.string()
        .matches(/^[0-2]$/, "Must be an integer between 0 and 2")
        .required('Release Due Date field is Required'),
});
