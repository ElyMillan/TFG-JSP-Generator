import React, { useMemo, useState } from 'react';
import { createTheme, ThemeProvider, CssBaseline, Box, Grid, Switch, FormGroup, FormControlLabel } from '@mui/material';
import Brightness7Icon from '@mui/icons-material/Brightness7'; // ícono del sol
import Brightness4Icon from '@mui/icons-material/Brightness4'; // ícono de la luna
import VerticalStepperForm from './components/VerticalStepperForm';
import Navbar from './layouts/Navbar';
import Footer from './layouts/Footer';

const darkThemeColors = {
  primary: {
    100: '#00ADB5',
    200: '#AAE3E2',
    300: '#fdf6fd',
  },
  secondary: {
    100: '#fdfdfd',
    200: '#fdfdfd',
    300: '#fdfdfd',
  },
  accent: {
    100: '#AC7DD2',
    200: '#fff4ff',
  },
  text: {
    100: '#EEEEEE',
    200: '#C5C5C5',
  },
  bg: {
    100: '#222831',
    200: '#393E46',
    300: '#454e59',
  }
};

const lightThemeColors = {
  primary: {
    100: '#005B99',
    200: '#4e88ca',
    300: '#b7e9ff',
  },
  accent: {
    100: '#FFD700',
    200: '#e9aa2b',
  },
  text: {
    100: '#333333',
    200: '#5c5c5c',
  },
  bg: {
    100: '#F5F5F5',
    200: '#ebebeb',
    300: '#c2c2c2',
  }
};

function App() {
  const [darkMode, setDarkMode] = useState(true);

  const theme = useMemo(() => createTheme({
    palette: {
      mode: darkMode ? 'dark' : 'light',
      primary: {
        main: darkMode ? darkThemeColors.primary[200] : lightThemeColors.primary[200],
      },
      background: {
        default: darkMode ? darkThemeColors.bg[100] : lightThemeColors.bg[100],
      },
      text: {
        primary: darkMode ? darkThemeColors.text[100] : lightThemeColors.text[100],
      }
    },
    components: {
      MuiCssBaseline: {
        styleOverrides: {
          body: {
            '& input:-webkit-autofill': {
              WebkitBoxShadow: `0 0 0 100px ${darkMode ? darkThemeColors.bg[100] : lightThemeColors.bg[100]} inset`,
              WebkitTextFillColor: darkMode ? darkThemeColors.text[100] : lightThemeColors.text[100],
              caretColor: darkMode ? darkThemeColors.text[100] : lightThemeColors.text[100],
            }
          }
        },
      },
      MuiInput: {
        styleOverrides: {
          input: {
            '&:-webkit-autofill': {
              WebkitBoxShadow: `0 0 0 100px ${darkMode ? darkThemeColors.bg[100] : lightThemeColors.bg[100]} inset`,
            }
          }
        }
      },
      MuiOutlinedInput: {
        styleOverrides: {
          input: {
            '&:-webkit-autofill': {
              WebkitBoxShadow: `0 0 0 100px ${darkMode ? darkThemeColors.bg[100] : lightThemeColors.bg[100]} inset`,
            }
          }
        }
      }
    }
  }), [darkMode]);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box>
        <Navbar darkMode={darkMode} />
      </Box>

      <Box sx={{ position: 'fixed', top: 42, right: 16, zIndex: 1300 }}>
        <FormGroup>
          <FormControlLabel
            control={
              <Switch
                checked={darkMode}
                onChange={() => setDarkMode(!darkMode)}
                icon={<Brightness7Icon />}
                checkedIcon={<Brightness4Icon />}
              />
            }
            label={darkMode ? 'Light' : 'Dark'}
          />
        </FormGroup>
      </Box>


      <Box sx={{ width: '100%', display: 'flex', alignItems: 'start', minHeight: '60vh', marginTop: '4rem' }}>
        <Grid container justifyContent="center" alignItems="center" sx={{ padding: '10px 20px', alignItems: 'center', lineHeight: '1.9' }}>
          <Grid item xs={10}>
            <VerticalStepperForm darkMode={darkMode} />
          </Grid>
        </Grid>
      </Box>

      <Box>
        <Footer darkMode={darkMode} />
      </Box>

    </ThemeProvider>
  );
}

export default App;
