import * as Yup from 'yup';

export const validationStep1 = Yup.object({
    email: Yup.string()
        .email("Invalid email address")
        .required("Email is required")
        .max(100, "Maximum 100 characters"),

    full_name: Yup.string()
        .matches(/^[\p{L}]+( [\p{L}]+)*$/u, "Name should only contain letters and single spaces between words")
        .required("Full name is required")
        .max(50, "Maximum 50 characters"),

    school_name: Yup.string()
        .matches(/^[\p{L}]+( [\p{L}]+)*$/u, "School name should only contain letters and single spaces between words")
        .required("School name is required")
        .max(50, "Maximum 50 characters"),
})