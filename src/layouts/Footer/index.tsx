import React from 'react'
import logo from "@assets/images/logo-upv.png";
import logoWhite from "@assets/images/logo-upv-white.png";
import './footer.css'; // Asumiendo que el archivo está en la misma carpeta que tu componente

const Footer = ({ darkMode }: { darkMode: boolean }) => {
    return (
        <>
            <footer className="bg-black">
                <div className="container footer-cols">
                    <div className="text-center">
                        <img src={darkMode ? logoWhite : logo} alt="Logo UPV" className="upv-logo" width="203" />
                    </div>
                    <hr style={{ backgroundColor: darkMode ? 'white' : 'black', border: 'none', height: '1px' }} />
                </div>

                <div className="social-links text-center" style={{ marginBottom: '2rem', marginTop: '1rem !important' }}>

                    <a href="https://www.linkedin.com/school/upv" title="Know our profile in Linkedin" target="_blank">
                        <svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
                            <path d="M5.07139 19.8188V7.13804H1.1272V19.8188H5.07139ZM3.07809 5.44162C4.35041 5.44162 5.36827 4.38135 5.36827 3.10903C5.36827 1.87912 4.35041 0.818848 3.07809 0.818848C1.84818 0.818848 0.830322 1.87912 0.830322 3.10903C0.830322 4.38135 1.84818 5.44162 3.07809 5.44162ZM19.8303 19.8188V12.8635C19.8303 9.47063 19.0669 6.84117 15.0803 6.84117C13.1718 6.84117 11.8995 7.90144 11.3482 8.87688H11.3058V7.13804H7.53122V19.8188H11.4754V13.5421C11.4754 11.888 11.7723 10.3188 13.8504 10.3188C15.8437 10.3188 15.8861 12.1849 15.8861 13.6693V19.8188H19.8303Z"></path>
                        </svg>
                    </a>
                    <a href="http://www.youtube.com/valenciaupv" title="All our videos in Youtube" target="_blank">
                        <svg width="24" height="17" viewBox="0 0 24 17" fill="currentColor">
                            <path d="M23.3258 2.86051C23.0735 1.86051 22.2746 1.06885 21.3075 0.818848C19.4994 0.318848 12.3513 0.318848 12.3513 0.318848C12.3513 0.318848 5.16122 0.318848 3.35317 0.818848C2.38608 1.06885 1.58718 1.86051 1.33489 2.86051C0.830322 4.61051 0.830322 8.36051 0.830322 8.36051C0.830322 8.36051 0.830322 12.0688 1.33489 13.8605C1.58718 14.8605 2.38608 15.6105 3.35317 15.8605C5.16122 16.3188 12.3513 16.3188 12.3513 16.3188C12.3513 16.3188 19.4994 16.3188 21.3075 15.8605C22.2746 15.6105 23.0735 14.8605 23.3258 13.8605C23.8303 12.0688 23.8303 8.36051 23.8303 8.36051C23.8303 8.36051 23.8303 4.61051 23.3258 2.86051ZM9.99668 11.7355V4.98551L15.9674 8.36051L9.99668 11.7355Z"></path>
                        </svg>
                    </a>

                </div>

                <div className="contact-links text-center">
                    <a href="https://goo.gl/maps/W6M95d45YoKFDhHA9" title="Find us on Google Map">
                        <svg width="16" height="17" viewBox="0 0 16 17" fill="currentColor">
                            <path d="M7.45312 15.1196C7.69922 15.5024 8.27344 15.5024 8.51953 15.1196C12.5117 9.37744 13.25 8.77588 13.25 6.64307C13.25 3.74463 10.8984 1.39307 8 1.39307C5.07422 1.39307 2.75 3.74463 2.75 6.64307C2.75 8.77588 3.46094 9.37744 7.45312 15.1196ZM8 8.83057C6.76953 8.83057 5.8125 7.87354 5.8125 6.64307C5.8125 5.43994 6.76953 4.45557 8 4.45557C9.20312 4.45557 10.1875 5.43994 10.1875 6.64307C10.1875 7.87354 9.20312 8.83057 8 8.83057Z"></path>
                        </svg>
                        <span>Camino de Vera, s/n. 46022 - Valencia</span>
                    </a>
                    <a href="tel:+34963877000" title="Call us by phone">
                        <svg width="15" height="15" viewBox="0 0 15 15" fill="currentColor">
                            <path d="M14.4805 1.07666L11.6367 0.42041C11.3359 0.338379 11.0078 0.502441 10.8711 0.803223L9.55859 3.86572C9.44922 4.13916 9.53125 4.43994 9.75 4.63135L11.418 5.99854C10.4336 8.07666 8.71094 9.82666 6.57812 10.8384L5.21094 9.17041C5.01953 8.95166 4.71875 8.86963 4.44531 8.979L1.38281 10.2915C1.08203 10.4282 0.945312 10.7563 1 11.0571L1.65625 13.9009C1.73828 14.2017 1.98438 14.3931 2.3125 14.3931C9.3125 14.3931 15 8.73291 15 1.70557C15 1.40479 14.7812 1.15869 14.4805 1.07666Z"></path>
                        </svg> +34 96 387 70 00 </a>

                </div>

                <div className="absolute-footer bg-white">
                    <div className="container">
                        <div className="links">
                            <a href="http://www.upv.es/contenidos/PORTRANSV4/" title="Our Transparency Page"> Transparency, </a>
                            <a href="http://www.upv.es/entidades/CYO/" title="Page with information on contracting in the UPV"> Perfil del contratante </a>
                            <a href="http://www.upv.es/otros/mapa-web-en.html" title="Site with the web map of UPV"> Site map </a>
                            <a href="legal/aviso-legal-upv-en.html" title="Legal information regarding the General Terms and Conditions that regulate the access, navigation and use of this Internet Portal"> Legal Notice </a>
                            <a href="/legal/politica-cookies-en.html" title="Information regarding the use of cookies on the website"> Cookies policy </a>
                            <a href="/legal/politica-privacidad-en.html" title="Information on data protection"> Política de privacidad </a>
                        </div>
                        <p className="copyright"> ©2024 Universitat Politècnica de València </p>
                    </div>
                </div>
            </footer>
        </>
    )
}

export default Footer