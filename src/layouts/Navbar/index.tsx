import { Grid, Typography, useMediaQuery, useTheme } from '@mui/material';
import logo from "@assets/images/logo-upv.png";
import logoWhite from "@assets/images/logo-upv-white.png";

const Navbar = ({ darkMode }: { darkMode: boolean }) => {
    const theme = useTheme();
    const isMobile = useMediaQuery(theme.breakpoints.down('lg'));

    return (
        <Grid container sx={{ padding: "1.5rem", marginBottom: "1rem", alignItems: 'center', position: 'relative', flexDirection: isMobile ? 'column' : 'row' }}>
            {/* Imagen del logo */}
            <Grid item sx={{ display: 'flex', justifyContent: isMobile ? 'center' : 'flex-start', width: isMobile ? '100%' : 'auto', marginBottom: isMobile ? '1rem' : 0 }}>
                <a href="/">
                    <img src={darkMode ? logoWhite : logo} alt="Logo UPV" className="upv-logo" style={{ width: isMobile ? '200px' : '250px' }} />
                </a>
            </Grid>

            {/* Título */}
            <Grid item sx={{ display: 'flex', justifyContent: 'center', width: isMobile ? '100%' : 'auto' }}>
                <Typography variant="h1" sx={{ fontSize: '2rem', textAlign: isMobile ? 'center' : 'center' }}>
                    JSP Instance Generator
                </Typography>
            </Grid>
        </Grid>
    );
}

export default Navbar;
