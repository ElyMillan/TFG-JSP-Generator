import React, { useEffect } from 'react';
import { Box, FormControl, Grid, TextField } from '@mui/material';
import { useFormik } from 'formik';
import * as Yup from 'yup';
import { validationStep1 } from './step1Validations';
import { FormData } from "./constants"

interface StepData {
    formData: FormData;
    setFormData: (formData: FormData) => void;
    isStepValid: boolean
    setStepValid: (isStepValid: boolean) => void;
    updateFormData: (data: Partial<FormData>, reset: boolean) => void;
}


const Step1 = ({ formData, setFormData, isStepValid, setStepValid, updateFormData }: StepData) => {

    const formik = useFormik({
        validateOnMount: true,
        initialValues: {
            email: formData.email,
            full_name: formData.full_name,
            school_name: formData.school_name,
        },
        validationSchema: validationStep1,
        onSubmit: async (values) => {
            console.log(values);
        },
        validateOnChange: true,
        validateOnBlur: true,
        enableReinitialize: true
    });


    useEffect(() => {
        if (!formik.isValid) {
            setStepValid(false)
        }
        if (formik.isValid) {
            setStepValid(true)
            updateFormData({
                full_name: formik.values.full_name,
                email: formik.values.email,
                school_name: formik.values.school_name,
            }, false);
        }
    }, [formik.values, formik.touched, formik.isValid]);


    return (
        <>
            <Box>
                <Grid container spacing={0}>
                    <Grid item xs={12} lg={12} sx={{ paddingX: '1rem' }}>
                        <FormControl fullWidth margin="normal" required sx={{ marginBottom: "1em !important" }}>
                            <TextField
                                name="email"
                                label="Email"
                                variant="standard"
                                fullWidth
                                margin="normal"
                                onChange={formik.handleChange}
                                onBlur={formik.handleBlur}
                                value={formik.values.email}
                                error={formik.touched.email && Boolean(formik.errors.email)}
                                helperText={formik.touched.email && formik.errors.email}
                                InputLabelProps={{
                                    shrink: true,
                                }}
                            />
                        </FormControl>
                    </Grid>


                    <Grid item xs={12} lg={12} sx={{ paddingX: '1rem' }}>
                        <FormControl fullWidth margin="normal" required sx={{ marginBottom: "1em !important" }}>
                            <TextField
                                name="full_name"
                                label="Full name"
                                variant="standard"
                                fullWidth
                                margin="normal"
                                onChange={formik.handleChange}
                                onBlur={formik.handleBlur}
                                value={formik.values.full_name}
                                error={formik.touched.full_name && Boolean(formik.errors.full_name)}
                                helperText={formik.touched.full_name && formik.errors.full_name}
                                InputLabelProps={{
                                    shrink: true,
                                }}
                            />
                        </FormControl>
                    </Grid>

                    <Grid item xs={12} lg={12} sx={{ paddingX: '1rem' }}>
                        <FormControl fullWidth margin="normal" required sx={{ marginBottom: "1em !important" }}>
                            <TextField
                                name="school_name"
                                label="School name"
                                variant="standard"
                                fullWidth
                                margin="normal"
                                onChange={formik.handleChange}
                                onBlur={formik.handleBlur}
                                value={formik.values.school_name}
                                error={formik.touched.school_name && Boolean(formik.errors.school_name)}
                                helperText={formik.touched.school_name && formik.errors.school_name}
                                InputLabelProps={{
                                    shrink: true,
                                }}
                            />
                        </FormControl>
                    </Grid>
                </Grid>
            </Box>
        </>
    );
}

export default Step1;
