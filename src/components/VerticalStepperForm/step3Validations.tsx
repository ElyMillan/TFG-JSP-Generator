import * as Yup from 'yup';

export const validationStep3 = Yup.object({
    pickle_file_output: Yup.boolean(),
    json_file_output: Yup.boolean(),
    dzn_file_output: Yup.boolean(),
    taillard_file_output: Yup.boolean(),
    single_folder_output: Yup.boolean(),
    custom_folder_name: Yup.string()
        .required("Custom folder name is required.")
        .matches(/^[^\\/:*?"<>|]+$/, {
            message: "Folder name can only contain alphanumeric characters, spaces, hyphens, and underscores."
        })
        .test(
            'validPlaceholders',
            'Invalid placeholder found. Only {size}, {jobs}, {machines}, {release_due_date}, {speed_scaling}, {seed}, {distribution} are allowed.',
            value => {
                const pattern = /\{([^}]+)\}/g;
                const validKeys = ['size', 'jobs', 'machines', 'release_due_date', 'speed_scaling', 'seed', 'distribution'];
                const matches = [...value.matchAll(pattern)];
                return matches.every(match => validKeys.includes(match[1]));
            }
        ),
    anyChecked: Yup.boolean().test(
        'checkOne',
        'You must select at least one output format',
        function () {
            return this.parent.pickle_file_output || this.parent.json_file_output || this.parent.dzn_file_output || this.parent.taillard_file_output;
        }
    ),
});
